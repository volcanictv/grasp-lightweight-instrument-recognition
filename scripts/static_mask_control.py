"""Control for the multi-frame refinement: does the gain need a tracker, or does viewing the same image region at other moments give it?

For every instrument that the registered runs track, the keyframe's own mask and box are applied unchanged to the neighbouring frames (no propagation at all), the four members of each classifier
seed classify those crops, and the result is written in the format of the registered tracked runs, so scripts/gtbox_sam_final_eval.py scores it with the same fusion, gate and metrics.
Two windows are derived from one pass: the past 20 frames (causal) and 10 frames on each side (non-causal).

Usage (titanxp, surgical environment, repo root):
    CUDA_VISIBLE_DEVICES=1 python scripts/static_mask_control.py --variant final
then for each window W in static_causal static_noncausal and seed S (42, 43, 44), tracked dir experiments/gtbox_sam/<variant>/tracked_<W>[_s43|_s44]:
    python scripts/gtbox_sam_final_eval.py --masks .../masks.pkl --logits .../logits_sS.npz --tracked-dir <dir> --tau 2.85e-4 --out docs/reports/gtbox_sam/<variant>_<W>_sS.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import build_track_frame_nums, crop_from_box
from evidential_seeds_e2e_eval import ORDER
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

SEEDS = (42, 43, 44)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="final")
    ap.add_argument("--past", type=int, default=20)
    ap.add_argument("--future", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="first N instruments only (smoke test)")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()
    dev = "cuda"
    D = REPO_ROOT / "experiments" / "gtbox_sam" / args.variant
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    frames_root = args.data_root / "frames-001" / "frames"
    masks = pickle.loads((D / "masks.pkl").read_bytes())
    idx = [e["index"] for e in json.loads((D / "tracked_causal_sam2" / "idx.json").read_text())["errors"]]
    if args.limit:
        idx = idx[: args.limit]
    print(f"{len(idx)} instruments", flush=True)

    ens = {}
    for s in SEEDS:
        cfg = yaml.safe_load((REPO_ROOT / f"configs/arms/ens4_N_official_s{s}.yaml").read_text())
        by_label = {}
        for m in cfg["members"]:
            net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(dev)
            net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=dev), strict=False)
            by_label[m["label"]] = (net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"], m["image_size"])
        ens[s] = [by_label[k] for k in ORDER]
    tf_by_key: dict[tuple, object] = {}  # one transform per distinct input: the members and seeds that share an input size and crop type share their transformed crops
    for s in SEEDS:
        for _net, tf, lb, size in ens[s]:
            tf_by_key.setdefault((size, lb), tf)

    out = {(w, s): {} for w in ("static_causal", "static_noncausal") for s in SEEDS}
    t0 = time.time()
    cache: dict[str, np.ndarray] = {}  # decoded frames, shared by the instruments of one keyframe (they have the same window)

    def decode(path: Path) -> np.ndarray:
        key = str(path)
        if key not in cache:
            if len(cache) > 80:
                cache.clear()
            cache[key] = np.array(Image.open(path).convert("RGB"))
        return cache[key]

    for n, i in enumerate(idx):
        file_name, _seg, box, _label = ds.instances[i]
        case, stem = file_name.split("/")
        nums, c = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.past, args.future)
        mask = mask_codec.decode(masks[i]).astype(bool)
        tensors: dict[tuple, list] = {}  # (image size, letterbox) -> the transformed crop of every frame; shared by the members and seeds that use the same input
        for fn in nums:
            frame = decode(frames_root / case / f"{fn:05d}.jpg")
            crops = {lb: crop_from_box(frame, mask, box, lb) for lb in (True, False)}
            for key, tf in tf_by_key.items():
                tensors.setdefault(key, []).append(tf(Image.fromarray(crops[key[1]])))
        batches = {key: torch.stack(v).to(dev) for key, v in tensors.items()}  # (frames, 3, size, size)
        per_seed = {}
        with torch.no_grad():
            for s in SEEDS:
                cols = [net(batches[(size, lb)]).float().cpu().numpy() for net, _tf, lb, size in ens[s]]  # each (frames, classes)
                per_seed[s] = np.stack(cols, axis=1)  # (frames, members, classes)
        for s in SEEDS:
            arr = per_seed[s]  # (frames, members, classes), frames in time order, keyframe at position c
            causal = arr[: c + 1]
            noncausal = arr[max(0, c - 10): c + 11]
            out[("static_causal", s)][f"det_{i}"] = causal
            out[("static_noncausal", s)][f"det_{i}"] = noncausal
        del tensors, batches, per_seed  # about 70 MB per instrument: free it before the next one
        if (n + 1) % 25 == 0:
            gc.collect()
            print(f"{n + 1}/{len(idx)}  {time.time() - t0:.0f}s  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)
    for (w, s), store in out.items():
        d = D / (f"tracked_{w}" + ("" if s == 42 else f"_s{s}"))
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "frames_shard0.npz", **store)
        print("wrote", d, len(store))


if __name__ == "__main__":
    main()
