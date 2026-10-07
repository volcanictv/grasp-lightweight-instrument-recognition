"""Builds the small test-data bundle for the A100 latency run on the RIT cluster: only what the benchmarks read, not the 14 GB dataset.

  * annotations/grasp_short-term_test.json            (instrument boxes and masks of the official test split)
  * the 63 evenly spaced test frames that scripts/benchmark_rt_segmenters.py uses (60 timed plus 3 warm-up)
  * the +-10-frame windows around the first 20 instruments of an index file, which scripts/benchmark_tracker_latency.py --stage propagate uses
  * e2e_keyframes.json: 32 evenly spaced test frames of the whole-pipeline runs (scripts/benchmark_pipeline_e2e.py), each kept with 20 past and 10 future frames
  * masks_subset.pkl: the saved real-time masks of the instruments in those 63 frames (classifier crop timing)
  * idx_tracker.json: the 20 instrument indices
  * MANIFEST.sha256: a checksum of every file, to verify the copy on the other side

Usage (titanxp, surgical environment, repo root):
    python scripts/build_rc_bench_subset.py --out ~/rc_bundle/subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np

from evaluate_temporal_track_ensemble import build_track_frame_nums
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data import splits


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--masks", type=Path, default=REPO_ROOT / "experiments" / "gtbox_sam" / "rt_tiny" / "masks.pkl")
    ap.add_argument("--tracker-idx", type=Path, default=REPO_ROOT / "experiments" / "official_causal" / "edgetam_armN" / "idx.json")
    ap.add_argument("--logits-rt", type=Path, default=REPO_ROOT / "experiments" / "gtbox_sam" / "rt_tiny" / "logits_s44.npz", help="seed 44 single-pass logits of the real-time masks")
    ap.add_argument("--logits-final", type=Path, default=REPO_ROOT / "experiments" / "gtbox_sam" / "final" / "logits_s44.npz", help="seed 44 single-pass logits of the final pipeline's masks")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--window", type=int, default=20, help="past frames kept for every window (the causal pipeline looks back 20)")
    ap.add_argument("--forward", type=int, default=10, help="future frames kept for every window (the non-causal pipeline uses +-10)")
    ap.add_argument("--e2e-keyframes", type=int, default=32, help="evenly spaced test frames for the whole-pipeline runs (30 timed plus 2 warm-up), each with its window")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    frames_root = args.data_root / "frames-001" / "frames"
    by_frame: dict[str, list[int]] = {}
    for i, inst in enumerate(ds.instances):
        by_frame.setdefault(inst[0], []).append(i)
    names = sorted(by_frame)
    pick = [names[i] for i in np.linspace(0, len(names) - 1, args.frames + args.warmup).round().astype(int)]  # the same rule as benchmark_rt_segmenters.py

    files: set[str] = set(pick)  # "CASE041/00035.jpg" style
    idx = [e["index"] for e in json.loads(args.tracker_idx.read_text())["errors"]][: args.n_instances]
    for i in idx:
        case, stem = ds.instances[i][0].split("/")
        nums, _c = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.window, args.forward)
        files.update(f"{case}/{n:05d}.jpg" for n in nums)

    def full_window(f: str) -> bool:  # a keyframe whose window really has the whole past and future, not the start or end of a case
        case, stem = f.split("/")
        n = int(stem.replace(".jpg", ""))
        return (frames_root / case / f"{n - args.window:05d}.jpg").exists() and (frames_root / case / f"{n + args.forward:05d}.jpg").exists()
    eligible = [f for f in names if full_window(f)]
    kf = [eligible[i] for i in np.linspace(0, len(eligible) - 1, args.e2e_keyframes).round().astype(int)]
    for f in kf:  # whole-pipeline keyframes, each with its past and future frames and the saved real-time masks of all its instruments
        case, stem = f.split("/")
        nums, _c = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.window, args.forward)
        files.update(f"{case}/{n:05d}.jpg" for n in nums)

    out = args.out
    (out / "GraSP" / "annotations").mkdir(parents=True, exist_ok=True)
    ann = splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS["test"]
    shutil.copy(ann, out / "GraSP" / "annotations" / ann.name)
    missing = []
    for f in sorted(files):
        src = frames_root / f
        dst = out / "GraSP" / "frames-001" / "frames" / f
        if not src.exists():
            missing.append(f)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    masks = pickle.loads(args.masks.read_bytes())
    keep = {i: masks[i] for f in pick for i in by_frame[f] if i in masks}
    (out / "masks_subset.pkl").write_bytes(pickle.dumps(keep))
    from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
    from surgical_ai.evaluation.evidential import variance_scores
    taus = {"tau_fold1": 2.85e-4}
    for tag, path in (("rt", args.logits_rt), ("final", args.logits_final)):  # the paper's own gate scores S1 of the test instruments, to reproduce its tracked share
        z = np.load(path)
        s1 = variance_scores(alpha_mix(lambda k: z["det_" + k], CONFIGS["four"]))["epistemic"]
        (out / f"gate_saved_{tag}.json").write_text(json.dumps({str(i): float(s1[i]) for i in range(len(s1)) if z["done"][i]}))
        budget = 539 if tag == "rt" else 833  # the paper's operating points: the 539 (real time) and 833 (final pipeline) most uncertain instruments of the whole test set
        taus[f"{tag}_{budget}"] = float(np.sort(s1[z["done"]])[::-1][budget - 1])
    (out / "gate_tau.json").write_text(json.dumps(taus))
    (out / "e2e_keyframes.json").write_text(json.dumps({"keyframes": kf, "warmup": 2}))
    (out / "idx_tracker.json").write_text(json.dumps({"errors": [{"index": int(i)} for i in idx]}))

    manifest = []
    total = 0
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.sha256":
            manifest.append(f"{sha256(p)}  {p.relative_to(out).as_posix()}")
            total += p.stat().st_size
    (out / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n")
    print(f"{len(files)} frames requested ({len(missing)} missing), {len(manifest)} files, {total / 1e6:.1f} MB; segmenter frames {len(pick)}, tracker instruments {len(idx)}")
    if missing:
        print("MISSING:", missing[:5])


if __name__ == "__main__":
    main()
