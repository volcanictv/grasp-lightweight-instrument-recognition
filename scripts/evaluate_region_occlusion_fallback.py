"""Targeted temporal fallback for Task B: only when an instance's mask
looks occlusion-fragmented (mask_utils.is_likely_occluded), track it a few
frames forward/backward with SAM2's video predictor and majority-vote the
ensemble's prediction across those extra views instead of trusting the
single annotated frame. Every other instance (97%+ of the test set) is
classified exactly as before, at no extra cost.

Built after directly testing why this is scoped narrowly, not applied
broadly: a temporal proof-of-concept (docs/DECISIONS.md, 2026-09-02) found
temporal context is a near-total fix for tissue-occlusion cases (40/41
nearby frames correct) but unreliable for aspect-ratio distortion (~1/3
correct, outvoted) and useless for genuine visual similarity (0/22). The
fragmentation signal fires almost exclusively on the occlusion cause
(validated directly against one known example of each of the three
causes before this script was written), so applying the temporal fallback
only when it fires avoids spending it on the two causes where it doesn't
help or actively hurts via added noise.

Usage:
    python scripts/evaluate_region_occlusion_fallback.py \\
        --classifier-a experiments/region_baseline_20260831-182451/best.pt \\
        --classifier-b experiments/region_letterbox_crop_20260902-152750/best.pt \\
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt \\
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from surgical_ai.data.mask_utils import decode_instance_mask, is_likely_occluded
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier-a", type=Path, required=True)
    parser.add_argument("--classifier-b", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--sam2-config", type=str, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--window", type=int, default=8, help="frames to track each direction")
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/sam2_fallback_frames"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def pad_to_square(crop: np.ndarray) -> np.ndarray:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - ch) // 2, (side - cw) // 2
    square[top : top + ch, left : left + cw] = crop
    return square


class EnsembleClassifier:
    def __init__(self, ckpt_a: Path, ckpt_b: Path, num_classes: int, image_size: int, device: torch.device):
        self.device = device
        self.transform = build_transforms(image_size, train=False)
        self.model_a = build_model("mobilenet_v3_small", num_classes=num_classes, pretrained=False, freeze_backbone=False).to(device)
        self.model_a.load_state_dict(torch.load(ckpt_a, map_location=device), strict=False)
        self.model_a.eval()
        self.model_b = build_model("mobilenet_v3_small", num_classes=num_classes, pretrained=False, freeze_backbone=False).to(device)
        self.model_b.load_state_dict(torch.load(ckpt_b, map_location=device), strict=False)
        self.model_b.eval()

    @torch.no_grad()
    def predict(self, crop: np.ndarray) -> np.ndarray:
        """crop: mask-multiplied RGB crop, native (non-letterboxed) shape."""
        t_a = self.transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        t_b = self.transform(Image.fromarray(pad_to_square(crop))).unsqueeze(0).to(self.device)
        probs_a = torch.softmax(self.model_a(t_a), dim=1)
        probs_b = torch.softmax(self.model_b(t_b), dim=1)
        return ((probs_a + probs_b) / 2).cpu().numpy()[0]


def track_and_vote(
    predictor, clf: EnsembleClassifier, data_root: Path, tmp_dir: Path,
    case: str, frame_num: int, gt_mask: np.ndarray, window: int,
) -> tuple[int, list[str]]:
    """Tracks the instance across a small window via SAM2 video propagation,
    classifies each tracked view, and majority-votes. Falls back to the
    original single-frame prediction (via the same classifier) if tracking
    yields too few usable frames.
    """
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    frame_nums = list(range(frame_num - window, frame_num + window + 1))
    kept_nums = []
    for i, fn in enumerate(frame_nums):
        src = data_root / "frames-001" / "frames" / case / f"{fn:05d}.jpg"
        if src.exists():
            shutil.copy(src, tmp_dir / f"{len(kept_nums):05d}.jpg")
            kept_nums.append(fn)
    if frame_num not in kept_nums:
        return None, []
    center_idx = kept_nums.index(frame_num)

    state = predictor.init_state(video_path=str(tmp_dir))
    predictor.add_new_mask(state, frame_idx=center_idx, obj_id=1, mask=gt_mask)

    tracked_masks = {}
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
        tracked_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=True):
        tracked_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()

    votes = []
    for local_idx in sorted(tracked_masks.keys()):
        frame_path = tmp_dir / f"{local_idx:05d}.jpg"
        if not frame_path.exists():
            continue
        mask = tracked_masks[local_idx]
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            continue
        frame = np.array(Image.open(frame_path).convert("RGB"))
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)
        probs = clf.predict(crop)
        votes.append(int(probs.argmax()))

    if len(votes) < 3:
        return None, []
    winner, _count = Counter(votes).most_common(1)[0]
    return winner, votes


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    ds = GraspRegionDataset(args.data_root, args.split, transform=None)
    class_names = ds.class_names_ordered()
    clf = EnsembleClassifier(args.classifier_a, args.classifier_b, len(class_names), args.image_size, device)

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    y_true, y_pred_baseline, y_pred_fallback = [], [], []
    n_triggered, n_fallback_used, n_fallback_changed_answer = 0, 0, 0

    instances = ds.instances if args.limit is None else ds.instances[: args.limit]
    for idx, (file_name, segmentation, (x, y, w, h), label_idx) in enumerate(instances):
        frame = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        height, width = frame.shape[:2]
        mask = decode_instance_mask(segmentation)

        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)
        baseline_pred = int(clf.predict(crop).argmax())

        y_true.append(label_idx)
        y_pred_baseline.append(baseline_pred)

        final_pred = baseline_pred
        if is_likely_occluded(mask):
            n_triggered += 1
            case, frame_str = file_name.split("/")
            frame_num = int(frame_str.replace(".jpg", ""))
            winner, _votes = track_and_vote(
                predictor, clf, args.data_root, args.tmp_dir, case, frame_num, mask.astype(bool), args.window,
            )
            if winner is not None:
                n_fallback_used += 1
                if winner != baseline_pred:
                    n_fallback_changed_answer += 1
                final_pred = winner

        y_pred_fallback.append(final_pred)

        if idx % 200 == 0:
            print(f"{idx}/{len(instances)}", flush=True)

    y_true = np.array(y_true)
    y_pred_baseline = np.array(y_pred_baseline)
    y_pred_fallback = np.array(y_pred_fallback)

    print(f"\ntriggered: {n_triggered}, fallback actually used (>=3 tracked frames): {n_fallback_used}, "
          f"fallback changed the answer: {n_fallback_changed_answer}")
    print(f"\nOverall accuracy -- baseline: {(y_pred_baseline == y_true).mean():.4f}")
    print(f"Overall accuracy -- with occlusion fallback: {(y_pred_fallback == y_true).mean():.4f}")

    triggered_mask = np.array([is_likely_occluded(decode_instance_mask(seg)) for _fn, seg, _b, _l in instances])
    print(f"\nOn the {triggered_mask.sum()} triggered instances only:")
    print(f"  baseline accuracy: {(y_pred_baseline[triggered_mask] == y_true[triggered_mask]).mean():.4f}")
    print(f"  fallback accuracy: {(y_pred_fallback[triggered_mask] == y_true[triggered_mask]).mean():.4f}")


if __name__ == "__main__":
    main()
