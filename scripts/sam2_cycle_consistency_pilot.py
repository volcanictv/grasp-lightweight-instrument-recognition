"""Tests cycle-consistency as a free, zero-vision-model triage signal for
SAM2 mask propagation quality (docs/DECISIONS.md, 2026-09-03): propagate
forward (source -> destination, same as sam2_propagation_pilot.py), then
re-seed the propagated destination mask and propagate *backward* across
the same frames back to the source. If forward tracking was clean, the
round-trip mask should land close to the real original source mask (high
IoU); if it drifted, the round trip should show it, with zero LLM cost --
just SAM2 running twice on frames already downloaded.

Uses the same 12 pilot pairs as sam2_propagation_pilot.py so the round-trip
IoU can be checked against the already-known real forward-destination IoU
(ground_truth_iou.json) for a genuine correlation check, not a guess.

Usage:
    python scripts/sam2_cycle_consistency_pilot.py \\
        --pairs sam2_pilot_pairs.json \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt \\
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/sam2_cycle_frames"))
    return parser.parse_args()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    from sam2.build_sam import build_sam2_video_predictor
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    pairs = json.loads(args.pairs.read_text())
    frames_root = args.data_root / "frames-001" / "frames"

    results = []
    for i, pair in enumerate(pairs):
        case = pair["case"]
        src_num = int(pair["src_frame"].split("/")[1].replace(".jpg", ""))
        dst_num = int(pair["dst_frame"].split("/")[1].replace(".jpg", ""))

        tmp_dir = args.tmp_dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        frame_nums = list(range(src_num, dst_num + 1))
        for local_idx, fn in enumerate(frame_nums):
            shutil.copy(frames_root / case / f"{fn:05d}.jpg", tmp_dir / f"{local_idx:05d}.jpg")
        last_idx = len(frame_nums) - 1

        src_mask_true = decode_instance_mask(pair["src_ann"]["segmentation"]).astype(bool)
        dst_mask_true = decode_instance_mask(pair["dst_ann"]["segmentation"]).astype(bool)

        # forward: source -> destination
        state = predictor.init_state(video_path=str(tmp_dir))
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=src_mask_true)
        forward_mask = None
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx == last_idx:
                forward_mask = (mask_logits[0, 0] > 0).cpu().numpy()

        if forward_mask is None or not forward_mask.any():
            print(f"[{i}] {case} {pair['class']} gap={pair['gap']}: forward propagation produced nothing, skipping cycle check")
            results.append({"index": i, "case": case, "class": pair["class"], "gap": pair["gap"],
                             "forward_iou_vs_true_dst": 0.0, "round_trip_iou_vs_true_src": 0.0, "forward_empty": True})
            continue

        forward_iou = mask_iou(forward_mask, dst_mask_true)

        # backward: re-seed at destination with the *propagated* mask (not ground truth), track back to source
        state = predictor.init_state(video_path=str(tmp_dir))
        predictor.add_new_mask(state, frame_idx=last_idx, obj_id=1, mask=forward_mask)
        round_trip_mask = None
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=True):
            if frame_idx == 0:
                round_trip_mask = (mask_logits[0, 0] > 0).cpu().numpy()

        round_trip_iou = mask_iou(round_trip_mask, src_mask_true) if round_trip_mask is not None else 0.0

        results.append({
            "index": i, "case": case, "class": pair["class"], "gap": pair["gap"],
            "forward_iou_vs_true_dst": forward_iou, "round_trip_iou_vs_true_src": round_trip_iou, "forward_empty": False,
        })
        print(f"[{i}] {case} {pair['class']} gap={pair['gap']}: forward_iou={forward_iou:.3f} round_trip_iou={round_trip_iou:.3f}")

    out_path = Path("sam2_cycle_consistency_results.json")
    out_path.write_text(json.dumps(results, indent=2))

    valid = [r for r in results if not r["forward_empty"]]
    if len(valid) >= 2:
        fwd = np.array([r["forward_iou_vs_true_dst"] for r in valid])
        rt = np.array([r["round_trip_iou_vs_true_src"] for r in valid])
        corr = np.corrcoef(fwd, rt)[0, 1]
        print(f"\ncorrelation between forward IoU (real quality) and round-trip IoU (free signal): {corr:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
