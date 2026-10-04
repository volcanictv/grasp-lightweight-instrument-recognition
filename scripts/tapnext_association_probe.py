"""Does TAPNext++ point tracking keep an instrument associated with a YOLO detection over 1 Hz gaps better than the IoU chain?

For each instrument (fold1, the gated top-20%): sample query points inside its ground-truth mask at the annotated frame,
track them backward through the past frames with TAPNext++ (online, one frame at a time, recurrent state), and in every past
frame ask which cached YOLO26s detection holds most of the visible tracked points. A frame counts as associated when at
least half of the visible points (and at least 8) fall inside one detection. The count is compared with the track length
the IoU chain (min-iou 0.1, coast 3, centre fallback) reached on the same instruments. No classifier is involved: this
measures association only.

Usage (titanxp, from ~/tapnext_venv):
    python scripts/tapnext_association_probe.py --n 60 --k 10 --out experiments/tapnext_probe/result.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import build_track_frame_nums
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--points", type=int, default=64)
    ap.add_argument("--checkpoint", type=Path, default=Path.home() / ".cache" / "tapnextpp" / "tapnextpp_ckpt.pt")
    ap.add_argument("--indices", type=Path, default=REPO_ROOT / "experiments" / "edgetam_causal" / "idx_top20.json")
    ap.add_argument("--detections", type=Path, default=REPO_ROOT / "experiments" / "yolo_causal" / "yolo26s" / "dets.pkl")
    ap.add_argument("--chain", type=Path, default=REPO_ROOT / "experiments" / "yolo_causal" / "yolo26s" / "fold1_tracked_shard0.json")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    frames_root = args.data_root / "frames-001" / "frames"
    ds = GraspRegionDataset(args.data_root, "fold1", letterbox=True)
    all_idx = [e["index"] for e in json.loads(args.indices.read_text())["errors"]]
    picked = all_idx[:: max(1, len(all_idx) // args.n)][: args.n]
    dets = pickle.loads(args.detections.read_bytes())
    chain = {r["index"]: r["track_len"] for r in json.loads(args.chain.read_text())["instances"]}

    t0 = time.time()
    model = TAPNextPP.from_checkpoint(args.checkpoint, device=args.device)
    model.warmup(args.points)
    print(f"model ready in {time.time() - t0:.1f} s", flush=True)

    rows, frame_ms = [], []
    for n, idx in enumerate(picked):
        file_name, seg, _box, _label = ds.instances[idx]
        case, stem = file_name.split("/")
        nums, center = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.k, 0)
        past = list(range(center - 1, -1, -1))  # newest past frame first: tracking runs backward in time
        if not past:
            continue
        gt = decode_instance_mask(seg).astype(bool)
        ys, xs = np.nonzero(gt)
        rng = np.random.default_rng(idx)
        pick = rng.choice(len(xs), size=min(args.points, len(xs)), replace=False)
        query = np.stack([xs[pick], ys[pick]], axis=1).astype(np.float32)

        def frame_at(local: int) -> np.ndarray:
            return cv2.imread(str(frames_root / case / f"{nums[local]:05d}.jpg"))

        _, _, state = model.track_frame(frame_at(center), query)
        associated, contiguous, alive = [], 0, True
        for local in past:
            torch.cuda.synchronize(args.device)
            t1 = time.perf_counter()
            pos, vis, state = model.track_frame(frame_at(local), state=state)
            torch.cuda.synchronize(args.device)
            frame_ms.append((time.perf_counter() - t1) * 1000)
            ok = False
            if vis.sum() >= 8:
                pts = np.round(pos[vis]).astype(int)
                h, w = gt.shape
                inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
                best = 0.0
                for rle in dets[str(frames_root / case / f"{nums[local]:05d}.jpg")]:
                    m = mask_codec.decode(rle).astype(bool)
                    frac = float(m[pts[inside, 1], pts[inside, 0]].sum()) / len(pts)
                    best = max(best, frac)
                ok = best >= 0.5
            associated.append(ok)
            if alive and ok:
                contiguous += 1
            else:
                alive = False
        rows.append({"index": idx, "past_frames": len(past), "tap_associated": int(sum(associated)), "tap_contiguous": contiguous,
                     "chain_past": min(args.k, chain.get(idx, 1) - 1)})
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(picked)}", flush=True)

    a = lambda key: float(np.mean([r[key] for r in rows]))
    summary = {"instruments": len(rows), "k": args.k, "mean_past_frames_available": a("past_frames"),
               "tap_associated_mean": a("tap_associated"), "tap_contiguous_mean": a("tap_contiguous"), "iou_chain_mean": a("chain_past"),
               "share_with_no_past_frame_tap": float(np.mean([r["tap_associated"] == 0 for r in rows])),
               "share_with_no_past_frame_chain": float(np.mean([r["chain_past"] == 0 for r in rows])),
               "tap_ms_per_frame_median": float(np.median(frame_ms)), "tap_ms_per_frame_p95": float(np.percentile(frame_ms, 95))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
