"""Does replacing the Mask R-CNN's own (weak) box-classifier label with the
validated Task B ensemble's prediction actually move AP50_segm, and by how
much -- tested directly rather than assumed. One model, one classifier
swap, no ensembling across segmentation models. Runs on the FULL official
test set (not a latency-benchmark sample) so the number is directly
comparable to the project's existing AP50_segm=0.8101 headline figure.

Same box+mask, same detection score (so ranking in the PR curve is
unaffected) -- only the category label changes, so any AP50_segm delta is
isolated to the classification swap alone.

Usage:
    python scripts/evaluate_classifier_swap_ap50.py \\
        --maskrcnn-checkpoint experiments/instance_segmentation_maskrcnn_official_20260901-180758/best.pt \\
        --classifier-a experiments/region_baseline_20260831-182451/best.pt \\
        --classifier-b experiments/region_letterbox_crop_20260902-152750/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from surgical_ai.data.detection_dataset import GraspDetectionDataset, build_detection_transforms
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.segmentation import evaluate_instance_ap50
from surgical_ai.models import build_model
from surgical_ai.models.detectors.registry import build_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maskrcnn-checkpoint", type=Path, required=True)
    parser.add_argument("--classifier-a", type=Path, required=True)
    parser.add_argument("--classifier-b", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def pad_to_square(crop: np.ndarray) -> np.ndarray:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - ch) // 2, (side - cw) // 2
    square[top : top + ch, left : left + cw] = crop
    return square


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    ds = GraspDetectionDataset(args.data_root, args.split, transform=build_detection_transforms(train=False))
    class_names = ds.class_names_ordered()

    maskrcnn = build_detector("maskrcnn_mobilenet_v3", num_classes=args.num_classes, pretrained=False).to(device)
    maskrcnn.load_state_dict(torch.load(args.maskrcnn_checkpoint, map_location=device))
    maskrcnn.eval()

    clf_a = build_model("mobilenet_v3_small", num_classes=args.num_classes, pretrained=False, freeze_backbone=False).to(device)
    clf_a.load_state_dict(torch.load(args.classifier_a, map_location=device), strict=False)
    clf_a.eval()
    clf_b = build_model("mobilenet_v3_small", num_classes=args.num_classes, pretrained=False, freeze_backbone=False).to(device)
    clf_b.load_state_dict(torch.load(args.classifier_b, map_location=device), strict=False)
    clf_b.eval()
    transform_a = build_transforms(args.image_size, train=False)
    transform_b = build_transforms(args.image_size, train=False)

    to_tensor = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    n_samples = len(ds.samples) if args.limit is None else min(args.limit, len(ds.samples))
    preds_original: list[list[tuple]] = []
    preds_swapped: list[list[tuple]] = []
    gt_instances: list[list[tuple]] = []

    for idx in range(n_samples):
        file_name, anns = ds.samples[idx]
        image = Image.open(ds.frames_root / file_name).convert("RGB")
        frame_np = np.array(image)
        height, width = frame_np.shape[:2]
        image_tensor = to_tensor(image).to(device)

        output = maskrcnn([image_tensor])[0]
        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        masks = output["masks"].cpu().numpy()[:, 0]

        img_original, img_swapped = [], []
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            if score < args.score_threshold:
                continue
            binary_mask = mask >= 0.5
            img_original.append((binary_mask, int(label) - 1, float(score)))

            x1, y1, x2, y2 = box
            x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
            x2i, y2i = min(width, int(round(x2))), min(height, int(round(y2)))
            if x2i <= x1i or y2i <= y1i:
                img_swapped.append((binary_mask, int(label) - 1, float(score)))  # degenerate box, keep original label
                continue
            crop_raw = frame_np[y1i:y2i, x1i:x2i]
            crop_mask = binary_mask[y1i:y2i, x1i:x2i]
            masked_crop = (crop_raw * crop_mask[:, :, None]).astype(np.uint8)

            tensor_a = transform_a(Image.fromarray(masked_crop)).unsqueeze(0).to(device)
            tensor_b = transform_b(Image.fromarray(pad_to_square(masked_crop))).unsqueeze(0).to(device)
            probs_a = torch.softmax(clf_a(tensor_a), dim=1)
            probs_b = torch.softmax(clf_b(tensor_b), dim=1)
            probs = ((probs_a + probs_b) / 2).cpu().numpy()[0]
            img_swapped.append((binary_mask, int(probs.argmax()), float(score)))

        preds_original.append(img_original)
        preds_swapped.append(img_swapped)

        gts = []
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if gt_mask.any():
                gts.append((gt_mask, ds._id_to_index[a["category_id"]]))
        gt_instances.append(gts)

        if idx % 100 == 0:
            print(f"{idx}/{n_samples}", flush=True)

    for _ in range(len(ds.samples) - n_samples):
        preds_original.append([])
        preds_swapped.append([])
        gt_instances.append([])

    result_original = evaluate_instance_ap50(preds_original, gt_instances, class_names)
    result_swapped = evaluate_instance_ap50(preds_swapped, gt_instances, class_names)

    print(f"\n=== Mask R-CNN's own classification head ===")
    print(f"AP50_segm = {result_original['map50']:.4f}")
    for name, ap in result_original["per_class_ap50"].items():
        print(f"  {name:<28} {ap:.4f}")

    print(f"\n=== Same boxes+masks+scores, Task B ensemble classification instead ===")
    print(f"AP50_segm = {result_swapped['map50']:.4f}")
    for name, ap in result_swapped["per_class_ap50"].items():
        print(f"  {name:<28} {ap:.4f}")

    print(f"\nDelta: {result_swapped['map50'] - result_original['map50']:+.4f}")


if __name__ == "__main__":
    main()
