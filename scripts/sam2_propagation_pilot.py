"""Pilot validation for the proposed Claude-as-mask-verifier pipeline
(new instructions, 2026-09-03): before trusting a vision-model judge on
genuinely unannotated frames, check whether it's even right on frames
where the real answer is already known.

For each (case, class, src_frame, dst_frame) pair -- both frames already
carry a real human annotation for the same instrument -- SAM2 propagates
the source frame's ground-truth mask forward across the real intervening
raw frames to the destination frame. The destination frame already has
its own real annotation, so the propagated mask's actual quality can be
scored objectively via IoU against it. That objective score is withheld
from the review step -- it exists purely to check the verifier's
judgment against, not to inform it.

Usage:
    python scripts/sam2_propagation_pilot.py \\
        --pairs sam2_pilot_pairs.json \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt \\
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \\
        --out-dir sam2_pilot_output
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
from PIL import Image, ImageDraw

from surgical_ai.data.mask_utils import decode_instance_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/sam2_pilot_frames"))
    parser.add_argument("--zoom", action="store_true", help="crop tight to the mask's bbox + padding instead of the full frame")
    parser.add_argument("--side-by-side", action="store_true", help="show the source frame's real mask next to the destination's propagated mask")
    return parser.parse_args()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def save_overlay(frame: np.ndarray, mask: np.ndarray, out_path: Path, color=(0, 255, 255), zoom: bool = False) -> None:
    """Outline-only: draws the mask's true boundary as a solid bright line,
    leaving every pixel inside and outside the mask untouched -- unlike a
    filled tint, this doesn't wash out the instrument's own texture/
    highlights, which a first pilot found was making a vision-model
    reviewer reject good masks as "flat, untextured blobs" (docs/DECISIONS.md,
    2026-09-03).

    `zoom=True` additionally crops to the mask's bounding box plus 60%
    padding on each side before drawing -- a second pilot found the
    reviewer sometimes locked onto a *different* similar-looking
    instrument elsewhere in a wide, cluttered surgical frame; a tight
    crop removes that distraction rather than asking for better judgment
    on a harder image."""
    from scipy import ndimage

    if not mask.any():
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "NO MASK PRODUCED", fill=(255, 0, 0))
        img.save(out_path)
        return

    boundary = mask & ~ndimage.binary_erosion(mask, iterations=2)
    overlay = frame.copy()
    overlay[boundary] = color

    if zoom:
        ys, xs = np.nonzero(mask)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        w, h = x1 - x0, y1 - y0
        pad_x, pad_y = int(w * 0.6) + 20, int(h * 0.6) + 20
        H, W = overlay.shape[:2]
        cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        cx1, cy1 = min(W, x1 + pad_x), min(H, y1 + pad_y)
        overlay = overlay[cy0:cy1, cx0:cx1]

    Image.fromarray(overlay).save(out_path)


def _draw_outline(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 255)) -> np.ndarray:
    from scipy import ndimage

    boundary = mask & ~ndimage.binary_erosion(mask, iterations=2)
    out = frame.copy()
    out[boundary] = color
    return out


def _crop_to_mask(frame: np.ndarray, mask: np.ndarray, pad_frac: float = 0.6) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = int(w * pad_frac) + 20, int(h * pad_frac) + 20
    H, W = frame.shape[:2]
    cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    cx1, cy1 = min(W, x1 + pad_x), min(H, y1 + pad_y)
    return frame[cy0:cy1, cx0:cx1]


def save_side_by_side(
    src_frame: np.ndarray, src_mask: np.ndarray, dst_frame: np.ndarray, dst_mask: np.ndarray, out_path: Path,
) -> None:
    """Left: the source frame's real, human-verified mask (proof of what the
    target instrument actually looks like). Right: the destination frame's
    propagated mask, zoomed the same way. Built to test whether showing the
    reviewer "the same object, verified" resolves cases where two similar
    instruments are both visible in a cluttered destination frame and the
    reviewer has no way to tell which one is the one actually being tracked
    (docs/DECISIONS.md, 2026-09-03 -- case 04 in the pilot)."""
    src_crop = _crop_to_mask(_draw_outline(src_frame, src_mask, color=(0, 255, 0)), src_mask)

    if dst_mask.any():
        dst_crop = _crop_to_mask(_draw_outline(dst_frame, dst_mask, color=(0, 255, 255)), dst_mask)
    else:
        dst_crop = dst_frame

    target_h = 420

    def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        w = max(1, int(img.shape[1] * h / img.shape[0]))
        return np.array(Image.fromarray(img).resize((w, h)))

    src_r, dst_r = resize_to_h(src_crop, target_h), resize_to_h(dst_crop, target_h)
    divider = np.full((target_h, 6, 3), (255, 255, 0), dtype=np.uint8)
    combined = np.concatenate([src_r, divider, dst_r], axis=1)

    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, src_r.shape[1], 22], fill=(0, 0, 0))
    draw.text((5, 4), "SOURCE (verified, this IS the instrument)", fill=(0, 255, 0))
    draw.rectangle([src_r.shape[1] + 6, 0, combined.shape[1], 22], fill=(0, 0, 0))
    draw.text((src_r.shape[1] + 11, 4), "DESTINATION (is the outline on the SAME object?)", fill=(0, 255, 255))
    img.save(out_path)


def main() -> None:
    args = parse_args()
    from sam2.build_sam import build_sam2_video_predictor
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    pairs = json.loads(args.pairs.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
            src_path = frames_root / case / f"{fn:05d}.jpg"
            shutil.copy(src_path, tmp_dir / f"{local_idx:05d}.jpg")

        src_mask = decode_instance_mask(pair["src_ann"]["segmentation"]).astype(bool)
        dst_mask_true = decode_instance_mask(pair["dst_ann"]["segmentation"]).astype(bool)

        state = predictor.init_state(video_path=str(tmp_dir))
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=src_mask)

        propagated_mask = None
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx == len(frame_nums) - 1:
                propagated_mask = (mask_logits[0, 0] > 0).cpu().numpy()

        if propagated_mask is None:
            print(f"[{i}] {case} {pair['class']} gap={pair['gap']}: propagation produced no mask at target frame, skipping")
            continue

        iou = mask_iou(propagated_mask, dst_mask_true)
        dst_frame_img = np.array(Image.open(frames_root / case / f"{dst_num:05d}.jpg").convert("RGB"))

        tag = f"{i:02d}_{case}_{pair['class'].replace(' ', '_')}_gap{pair['gap']}"
        overlay_path = args.out_dir / f"{tag}_propagated.png"
        if args.side_by_side:
            src_frame_img = np.array(Image.open(frames_root / case / f"{src_num:05d}.jpg").convert("RGB"))
            save_side_by_side(src_frame_img, src_mask, dst_frame_img, propagated_mask, overlay_path)
        else:
            save_overlay(dst_frame_img, propagated_mask, overlay_path, zoom=args.zoom)

        results.append({
            "index": i, "case": case, "class": pair["class"], "gap": pair["gap"],
            "src_frame": pair["src_frame"], "dst_frame": pair["dst_frame"],
            "overlay_path": str(overlay_path.name), "true_iou": iou,
        })
        print(f"[{i}] {case} {pair['class']} gap={pair['gap']}: IoU={iou:.3f} -> {overlay_path.name}")

    (args.out_dir / "ground_truth_iou.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} overlays + ground_truth_iou.json to {args.out_dir}")


if __name__ == "__main__":
    main()
