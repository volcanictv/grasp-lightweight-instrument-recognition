"""Fine-tune SAM2's mask decoder on GraSP (accuracy-first priority,
docs/DECISIONS.md 2026-09-01). Zero-shot SAM2 (box-prompted, frozen)
underperformed our own fine-tuned Mask R-CNN (AP50_segm 0.769 vs 0.8101) --
the natural follow-up flagged there but not yet tried: keep SAM2's image
encoder frozen (it's already an excellent, general-purpose feature
extractor; retraining a Hiera-Large backbone on 2324 images would be both
wasteful and prone to catastrophic forgetting) and fine-tune only the
lightweight mask decoder on GraSP's own instrument masks. Standard, widely-
documented SAM/SAM2 adaptation recipe, not novel.

Training uses ground-truth boxes as prompts (the decoder should learn to
produce a good mask given an accurate box, which is the actual supervision
available); final evaluation uses the real box detector's own predicted
boxes, matching how this would actually be deployed and matching every
other evaluation in this project.

Neither `SAM2ImagePredictor.predict()` (detaches outputs to numpy) nor its
internal `._predict()` (decorated `@torch.no_grad()`, unconditionally) can
be used for training -- both are inference-only by construction. This
module's `differentiable_predict()` replicates `_predict()`'s box-prompt
logic (verified against sam2/sam2_image_predictor.py) without that
decorator, calling `sam_prompt_encoder`/`sam_mask_decoder` directly so
gradients reach the mask decoder -- the standard way SAM/SAM2 fine-tuning
tutorials handle this, not an unsupported hack.

Usage:
    python scripts/finetune_sam2_decoder.py \\
        experiments/detection_weighted_loss_20260901-001115/best.pt \\
        ~/sam2/checkpoints/sam2.1_hiera_large.pt configs/sam2.1/sam2.1_hiera_l.yaml \\
        --data-root ./GraSP --device cuda:0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

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
    parser.add_argument("detector_checkpoint", type=Path, help="Box detector used for final evaluation only.")
    parser.add_argument("sam2_checkpoint", type=Path)
    parser.add_argument("sam2_config", type=str)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSP")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30, help="Ceiling, not a target -- patience-based early stopping.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-instances-per-image", type=int, default=6, help="Cap for memory; GraSP frames rarely exceed this.")
    parser.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap train/val images.")
    return parser.parse_args()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum()
    union = prob.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


def differentiable_predict(predictor, box: np.ndarray) -> torch.Tensor:
    """Single-box mask prediction, gradient-preserving. Replicates
    SAM2ImagePredictor._predict's box-prompt path exactly (box -> two
    corner "points" with labels 2/3, prompt encoder, mask decoder,
    upsample to original resolution) without its `@torch.no_grad()`
    decorator. Single box only (no batching -- `repeat_image=False`
    always, matching `_predict`'s own `batched_mode` for one box).

    Returns mask logits, shape (1, 1, H, W) at the original image size.
    """
    _mask_input, _unnorm_coords, _labels, unnorm_box = predictor._prep_prompts(None, None, box, None, True)
    box_coords = unnorm_box.reshape(-1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
    box_labels = box_labels.repeat(box_coords.size(0), 1)

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(box_coords, box_labels), boxes=None, masks=None,
    )
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res_masks, _iou_predictions, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    return predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])


def run_epoch(predictor, dataset, device: torch.device, optimizer=None, max_instances: int = 6) -> float:
    """One pass over `dataset`. If `optimizer` is given, trains the mask
    decoder (GT-box prompts); otherwise computes the same loss under
    torch.no_grad() as a cheap validation signal for early stopping.
    """
    train = optimizer is not None
    predictor.model.sam_mask_decoder.train(train)
    total_loss, n = 0.0, 0

    for file_name, anns in dataset.samples:
        boxes, masks = [], []
        for a in anns[:max_instances]:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            native_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if not native_mask.any():
                continue
            boxes.append([x, y, x + w, y + h])
            masks.append(native_mask)
        if not boxes:
            continue

        image = Image.open(dataset.frames_root / file_name).convert("RGB")
        image_np = np.array(image)
        with torch.no_grad():
            predictor.set_image(image_np)

        with torch.set_grad_enabled(train):
            for box, gt_mask in zip(boxes, masks):
                logits = differentiable_predict(predictor, np.array(box))
                gt_tensor = torch.from_numpy(gt_mask).float().to(device).unsqueeze(0).unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(logits, gt_tensor) + dice_loss(logits, gt_tensor)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item()
                n += 1

    return total_loss / max(n, 1)


def evaluate_final(predictor, detector, val_ds, occlusion_fractions, class_names, device, score_threshold: float = 0.5) -> None:
    predictor.model.sam_mask_decoder.eval()
    to_tensor = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    predictions: list[list[tuple]] = []
    gt_instances: list[list[tuple]] = []

    for file_name, anns in val_ds.samples:
        image = Image.open(val_ds.frames_root / file_name).convert("RGB")
        image_np = np.array(image)
        with torch.no_grad():
            det_out = detector([to_tensor(image).to(device)])[0]
        boxes = det_out["boxes"].cpu().numpy()
        labels = det_out["labels"].cpu().numpy()
        scores = det_out["scores"].cpu().numpy()
        keep = scores >= score_threshold

        image_preds = []
        if keep.any():
            with torch.no_grad():
                predictor.set_image(image_np)
                for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                    mask_input, _uc, _l, unnorm_box = predictor._prep_prompts(None, None, box, None, True)
                    pred_masks, _iou, _lr = predictor._predict(
                        None, None, unnorm_box, mask_input, multimask_output=False, return_logits=True
                    )
                    image_preds.append((pred_masks[0, 0].cpu().numpy() > 0, int(label) - 1, float(score)))
        predictions.append(image_preds)

        gts = []
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if gt_mask.any():
                gts.append((gt_mask, val_ds._id_to_index[a["category_id"]]))
        gt_instances.append(gts)

    ap50 = evaluate_instance_ap50(predictions, gt_instances, class_names)
    counts = {"isolated": 0, "light": 0, "heavy": 0}
    hits = {"isolated": 0, "light": 0, "heavy": 0}
    for image_idx, (_file_name, anns) in enumerate(val_ds.samples):
        preds = [(m, lbl, s) for m, lbl, s in predictions[image_idx] if s >= 0.5]
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
    logger.info("FINAL: AP50_segm=%.4f  occlusion recall: %s", ap50["map50"], recall_str)
    for name, ap in ap50["per_class_ap50"].items():
        logger.info("  %s: %.3f", name, ap)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    train_split, val_split = splits.resolve_train_val_split("official")
    train_ds = GraspDetectionDataset(args.data_root, train_split, include_masks=True)
    val_ds = GraspDetectionDataset(args.data_root, val_split, include_masks=True)
    class_names = val_ds.class_names_ordered()
    if args.limit:
        train_ds.samples = train_ds.samples[: args.limit]
        val_ds.samples = val_ds.samples[: args.limit]
    occlusion_fractions = compute_occlusion_fractions(val_ds)

    predictor = SAM2ImagePredictor(build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=str(device)))
    predictor.model.image_encoder.requires_grad_(False)
    predictor.model.sam_prompt_encoder.requires_grad_(False)
    # mask decoder left trainable (default requires_grad=True)

    optimizer = torch.optim.Adam(predictor.model.sam_mask_decoder.parameters(), lr=args.lr)

    detector = build_detector("fasterrcnn_mobilenet_v3", num_classes=len(class_names), pretrained=False).to(device)
    detector.load_state_dict(torch.load(args.detector_checkpoint, map_location=device))
    detector.eval()

    run_id = f"sam2_decoder_finetune_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.experiments_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "decoder_best.pt"

    best_val_loss = float("inf")
    epochs_since_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(predictor, train_ds, device, optimizer=optimizer, max_instances=args.max_instances_per_image)
        val_loss = run_epoch(predictor, val_ds, device, optimizer=None, max_instances=args.max_instances_per_image)
        logger.info("epoch %d/%d train_loss=%.4f val_loss=%.4f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(predictor.model.sam_mask_decoder.state_dict(), checkpoint_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                logger.info("early stopping: no val_loss improvement in %d epochs (best=%.4f)", args.patience, best_val_loss)
                break

    predictor.model.sam_mask_decoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
    evaluate_final(predictor, detector, val_ds, occlusion_fractions, class_names, device)
    logger.info("decoder checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
