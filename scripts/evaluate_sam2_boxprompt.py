"""Zero-shot SAM2 box-prompted segmentation, official test set (accuracy-first
priority, docs/DECISIONS.md 2026-09-01). Known, published direction (Zero-Shot
Surgical Tool Segmentation Using SAM2, arXiv 2408.01648; npj Digital Surgery's
SAM surgical-video evaluation), not something invented here.

Pipeline: our already-trained box detector (`fasterrcnn_mobilenet_v3`,
AP50_box 0.831-0.837) finds and classifies each instrument -- box, label, and
score come entirely from that model, unchanged. Each predicted box is then
used as a prompt into SAM2 (frozen, zero-shot, no fine-tuning) to produce the
mask, replacing this project's own from-scratch/fine-tuned mask head with a
foundation model's segmentation decoder. No new training in this script.

Requires the `sam2` package installed from facebookresearch/sam2 and a
downloaded checkpoint (see docs/DECISIONS.md for the exact commands used).

Usage:
    python scripts/evaluate_sam2_boxprompt.py \\
        experiments/detection_weighted_loss_20260901-001115/best.pt \\
        ~/sam2/checkpoints/sam2.1_hiera_large.pt ~/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \\
        --data-root ./GraSP --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data import splits  # noqa: E402
from surgical_ai.data.detection_dataset import GraspDetectionDataset  # noqa: E402
from surgical_ai.data.mask_utils import decode_instance_mask  # noqa: E402
from surgical_ai.evaluation.detection import compute_occlusion_fractions  # noqa: E402
from surgical_ai.evaluation.segmentation import evaluate_instance_ap50, mask_iou  # noqa: E402
from surgical_ai.models.detectors.registry import build_detector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detector_checkpoint", type=Path)
    parser.add_argument("sam2_checkpoint", type=Path)
    parser.add_argument("sam2_config", type=str, help="Config name as SAM2 expects it, e.g. configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Detector box confidence to bother prompting SAM2 with.")
    parser.add_argument("--recall-score-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _, val_split = splits.resolve_train_val_split("official")
    val_ds = GraspDetectionDataset(args.data_root, val_split, include_masks=True)
    class_names = val_ds.class_names_ordered()
    occlusion_fractions = compute_occlusion_fractions(val_ds)

    detector = build_detector("fasterrcnn_mobilenet_v3", num_classes=len(class_names), pretrained=False).to(device)
    detector.load_state_dict(torch.load(args.detector_checkpoint, map_location=device))
    detector.eval()

    sam2_predictor = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=str(device)))

    n_samples = len(val_ds.samples) if args.limit is None else min(args.limit, len(val_ds.samples))
    predictions: list[list[tuple]] = []
    gt_instances: list[list[tuple]] = []

    to_tensor = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    for idx in range(n_samples):
        file_name, anns = val_ds.samples[idx]
        image = Image.open(val_ds.frames_root / file_name).convert("RGB")
        image_np = np.array(image)

        with torch.no_grad():
            det_out = detector([to_tensor(image).to(device)])[0]
        boxes = det_out["boxes"].cpu().numpy()
        labels = det_out["labels"].cpu().numpy()
        scores = det_out["scores"].cpu().numpy()
        keep = scores >= args.score_threshold

        image_preds = []
        if keep.any():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                sam2_predictor.set_image(image_np)
                for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                    masks, mask_scores, _ = sam2_predictor.predict(box=box, multimask_output=False)
                    image_preds.append((masks[0] > 0.5, int(label) - 1, float(score)))
        predictions.append(image_preds)

        gts = []
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if gt_mask.any():
                gts.append((gt_mask, val_ds._id_to_index[a["category_id"]]))
        gt_instances.append(gts)

        if idx % 50 == 0:
            print(f"{idx}/{n_samples}", flush=True)

    for _ in range(len(val_ds.samples) - n_samples):
        predictions.append([])
        gt_instances.append([])

    ap50 = evaluate_instance_ap50(predictions, gt_instances, class_names)

    counts = {"isolated": 0, "light": 0, "heavy": 0}
    hits = {"isolated": 0, "light": 0, "heavy": 0}
    for image_idx in range(len(val_ds.samples)):
        _file_name, anns = val_ds.samples[image_idx]
        preds = [(m, lbl, s) for m, lbl, s in predictions[image_idx] if s >= args.recall_score_threshold]
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if not gt_mask.any():
                continue
            gt_label = val_ds._id_to_index[a["category_id"]]
            frac = occlusion_fractions.get(a["id"], 0.0)
            bucket = "isolated" if frac <= 0.0 else ("heavy" if frac > 0.5 else "light")
            counts[bucket] += 1
            if any(lbl == gt_label and mask_iou(gt_mask, m) >= 0.5 for m, lbl, _s in preds):
                hits[bucket] += 1

    recall_str = ", ".join(f"{b}={hits[b] / counts[b] if counts[b] else float('nan'):.3f}" for b in ("isolated", "light", "heavy"))
    print(f"\nAP50_segm={ap50['map50']:.4f}")
    print(f"occlusion recall: {recall_str}")
    print("per-class AP50:")
    for name, ap in ap50["per_class_ap50"].items():
        print(f"  {name}: {ap:.3f}")


if __name__ == "__main__":
    main()
