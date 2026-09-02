"""Train a Task A (multilabel_frame), Task B (region_classification),
detection, or segmentation model from a config -- `config["task"]` selects
which.

Usage:
    python scripts/train.py configs/baseline_frozen.yaml [--data-root PATH] [--device cuda:0]

Every run writes experiments/<run_id>/manifest.json, best.pt, metrics.json,
and figures/. No hyperparameters as CLI flags, per CLAUDE.md — only paths and
device change between invocations without editing the config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data import splits  # noqa: E402
from surgical_ai.data.dataset import GraspMultiLabelDataset  # noqa: E402
from surgical_ai.data.copy_paste import CopyPasteDetectionDataset  # noqa: E402
from surgical_ai.data.detection_dataset import (  # noqa: E402
    GraspDetectionDataset,
    build_detection_transforms,
    collate_fn,
)
from surgical_ai.data.mask_utils import decode_instance_mask  # noqa: E402
from surgical_ai.data.region_dataset import GraspRegionDataset  # noqa: E402
from surgical_ai.data.segmentation_copy_paste import SegmentationCopyPasteDataset  # noqa: E402
from surgical_ai.data.segmentation_dataset import (  # noqa: E402
    GraspSegmentationDataset,
)
from surgical_ai.data.segmentation_dataset import collate_fn as segmentation_collate_fn  # noqa: E402
from surgical_ai.data.segmentation_targets import downsample_mask_nearest  # noqa: E402
from surgical_ai.data.transforms import build_transforms  # noqa: E402
from surgical_ai.evaluation.detection import (  # noqa: E402
    compute_occlusion_fractions,
    dataset_to_coco_gt,
    evaluate_occlusion_stratified_recall,
)
from surgical_ai.evaluation.segmentation import (  # noqa: E402
    decode_instances,
    evaluate_instance_ap50,
    evaluate_occlusion_stratified_recall_segm,
    mask_iou,
)
from surgical_ai.models import build_model  # noqa: E402
from surgical_ai.models.detectors.registry import build_detector  # noqa: E402
from surgical_ai.models.detectors.weighted_loss import (  # noqa: E402
    apply_class_weighted_detection_loss,
)
from surgical_ai.models.segmenters.registry import build_segmenter  # noqa: E402
from surgical_ai.training.losses import (  # noqa: E402
    build_loss,
    compute_class_weights,  # also used for detection's box-classification loss
    compute_pos_weight,
)
from surgical_ai.training.samplers import build_sampler  # noqa: E402
from surgical_ai.training.trainer import (  # noqa: E402
    collect_detections,
    count_parameters,
    evaluate,
    evaluate_region,
    fit,
    fit_detection,
    fit_segmentation,
)
from surgical_ai.utils import visualization as viz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")),
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, training_config: dict) -> torch.optim.Optimizer:
    """Single LR by default. If training.backbone_lr is set (PROJECT_SPEC.md
    Sec.6's discriminative-LR recommendation for fine-tuning), splits params
    into a backbone group and a head group using `model.head` -- every
    registry builder in models/classifiers/ sets this to the replaced
    classification head submodule, so this is backbone-agnostic rather than
    matching architecture-specific attribute names.
    """
    lr = training_config["lr"]
    backbone_lr = training_config.get("backbone_lr")
    if backbone_lr is None:
        return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

    head_param_ids = {id(p) for p in model.head.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_param_ids]
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids
    ]
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": lr},
        ]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "note": "not a git repository"}


def gpu_info(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"device": "cpu"}
    return {
        "name": torch.cuda.get_device_name(device),
        "driver_capability": torch.cuda.get_device_capability(device),
        "torch_cuda_build": torch.version.cuda,
    }


def _setup_multilabel_task(config: dict, args: argparse.Namespace, device: torch.device):
    train_split, val_split = splits.resolve_train_val_split(config["data"]["split"])
    image_size = config["data"]["image_size"]
    augmentation = config["data"].get("augmentation", "default")

    train_ds = GraspMultiLabelDataset(
        args.data_root, train_split,
        transform=build_transforms(image_size, train=True, augmentation=augmentation),
    )
    val_ds = GraspMultiLabelDataset(
        args.data_root, val_split, transform=build_transforms(image_size, train=False)
    )
    class_names = train_ds.class_names_ordered()

    sampling = config["data"].get("sampling", "none")
    sampler = build_sampler(sampling, train_ds.samples)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=args.num_workers,
    )

    label_counts = torch.zeros(len(class_names))
    for _, label in train_ds.samples:
        label_counts += label

    pos_weight = None
    if config["loss"].get("class_weights", False):
        pos_weight = compute_pos_weight(label_counts, len(train_ds)).to(device)
    loss_fn = build_loss(config["loss"], pos_weight=pos_weight)

    return train_loader, val_loader, class_names, loss_fn, evaluate


def _setup_region_task(config: dict, args: argparse.Namespace, device: torch.device):
    train_split, val_split = splits.resolve_train_val_split(config["data"]["split"])
    image_size = config["data"]["image_size"]
    augmentation = config["data"].get("augmentation", "default")

    letterbox = config["data"].get("letterbox_crop", False)
    train_ds = GraspRegionDataset(
        args.data_root, train_split,
        transform=build_transforms(image_size, train=True, augmentation=augmentation),
        letterbox=letterbox,
    )
    val_ds = GraspRegionDataset(
        args.data_root, val_split, transform=build_transforms(image_size, train=False),
        letterbox=letterbox,
    )
    class_names = train_ds.class_names_ordered()

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=args.num_workers,
    )

    label_counts = torch.zeros(len(class_names))
    for _, _, _, label_idx in train_ds.instances:
        label_counts[label_idx] += 1

    class_weight = None
    if config["loss"].get("class_weights", False):
        class_weight = compute_class_weights(label_counts).to(device)
    loss_fn = build_loss(config["loss"], class_weight=class_weight)

    return train_loader, val_loader, class_names, loss_fn, evaluate_region


def build_detection_optimizer(
    model: torch.nn.Module, training_config: dict
) -> torch.optim.Optimizer:
    """Defaults to Adam (this project's convention everywhere else). SGD is
    the standard torchvision detection-recipe optimizer (momentum 0.9,
    weight_decay 5e-4 -- torchvision's own reference training script
    defaults) -- available via `training.optimizer: sgd` as a one-variable
    ablation against the Adam default, per the open question flagged in
    docs/DECISIONS.md.
    """
    params = (p for p in model.parameters() if p.requires_grad)
    opt_name = training_config.get("optimizer", "adam")
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=training_config["lr"])
    if opt_name == "sgd":
        return torch.optim.SGD(
            params, lr=training_config["lr"], momentum=0.9, weight_decay=5e-4
        )
    raise ValueError(f"unknown training.optimizer '{opt_name}'. Valid: adam, sgd")


def run_detection_training(config: dict, args: argparse.Namespace, device: torch.device) -> None:
    """Self-contained detection training path -- kept separate from the
    classification tasks' shared code below rather than threaded through
    it, since a detector's model interface (dict-of-losses in train mode,
    no external loss_fn, mAP-based checkpoint selection, no confusion
    matrix) diverges enough that forcing it through the same branches would
    obscure both paths rather than clarify either.
    """
    train_split, val_split = splits.resolve_train_val_split(config["data"]["split"])

    # "none" default matches the original Milestone 8 baseline's actual
    # behavior (predates this parameter existing) -- keeps old configs that
    # don't set data.augmentation reproducible. See data/detection_dataset.py.
    augmentation = config["data"].get("augmentation", "none")
    train_ds = GraspDetectionDataset(
        args.data_root, train_split,
        transform=build_detection_transforms(train=True, augmentation=augmentation),
    )
    val_ds = GraspDetectionDataset(
        args.data_root, val_split, transform=build_detection_transforms(train=False)
    )
    class_names = train_ds.class_names_ordered()
    coco_gt_val = dataset_to_coco_gt(val_ds)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    if config.get("loss", {}).get("class_weights", False):
        label_counts = torch.zeros(len(class_names))
        for _file_name, anns in train_ds.samples:
            for a in anns:
                label_counts[train_ds._id_to_index[a["category_id"]]] += 1
        # index 0 = background, weight 1.0 (not part of the imbalance this
        # is fixing); foreground classes get the same balanced formula
        # Task B's classifier uses (training/losses.py::compute_class_weights).
        class_weight = torch.cat([torch.tensor([1.0]), compute_class_weights(label_counts)])
        apply_class_weighted_detection_loss(class_weight.to(device))

    copy_paste_cfg = config["data"].get("copy_paste", {})
    train_loader_ds = train_ds
    if copy_paste_cfg.get("enabled", False):
        train_loader_ds = CopyPasteDetectionDataset(
            train_ds,
            paste_prob=copy_paste_cfg.get("paste_prob", 0.5),
            max_pastes=copy_paste_cfg.get("max_pastes", 2),
            rare_classes=copy_paste_cfg.get("rare_classes"),
            occlusion_bias=copy_paste_cfg.get("occlusion_bias", 0.7),
            seed=config["training"]["seed"],
        )
    train_loader = torch.utils.data.DataLoader(
        train_loader_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = build_detector(
        config["model"]["name"], num_classes=len(class_names),
        pretrained=config["model"]["pretrained"],
    ).to(device)
    trainable, total = count_parameters(model)
    optimizer = build_detection_optimizer(model, config["training"])

    run_id = f"{args.config.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.experiments_dir / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"

    start = time.time()
    history, best_metrics = fit_detection(
        model, train_loader, val_loader, optimizer, class_names, device,
        epochs=config["training"]["epochs"], checkpoint_path=checkpoint_path,
        coco_gt_val=coco_gt_val, patience=config["training"].get("patience"),
    )
    duration_sec = time.time() - start

    # Reload the best epoch's weights -- `model` in memory is whatever the
    # last epoch trained to, which isn't necessarily the checkpointed best
    # one, and the occlusion analysis below needs the actual best model.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    viz.plot_curve(history.train_loss, "train loss", "Train loss", figures_dir / "loss_curve.png")
    viz.plot_training_curves(
        history.val_map50, history.val_map50_95, "mAP", "Val mAP@50 vs. mAP@50:95",
        figures_dir / "map_curve.png",
    )

    occlusion_fractions = compute_occlusion_fractions(val_ds)
    val_predictions = collect_detections(model, val_loader, device)
    occlusion_recall = evaluate_occlusion_stratified_recall(
        val_ds, val_predictions, occlusion_fractions
    )
    print(occlusion_recall.to_markdown())

    split_paths = {
        train_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[train_split],
        val_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[val_split],
    }
    manifest = {
        "run_id": run_id,
        "config": config,
        "config_path": str(args.config),
        "git": git_info(),
        "seed": config["training"]["seed"],
        "split_files": {
            name: {"path": str(path), "sha256": sha256_of(path)}
            for name, path in split_paths.items()
        },
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
        "gpu": gpu_info(device),
        "trainable_params": trainable,
        "total_params": total,
        "wall_clock_seconds": duration_sec,
        "best_checkpoint": str(checkpoint_path),
        "final_metrics": {
            "map50": best_metrics.map50,
            "map50_95": best_metrics.map50_95,
            "per_class_ap50": best_metrics.per_class_ap50,
            "occlusion_stratified_recall": {
                "n_isolated": occlusion_recall.n_isolated,
                "n_light": occlusion_recall.n_light,
                "n_heavy": occlusion_recall.n_heavy,
                "recall_isolated": occlusion_recall.recall_isolated,
                "recall_light": occlusion_recall.recall_light,
                "recall_heavy": occlusion_recall.recall_heavy,
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "metrics_table.md").write_text(
        best_metrics.to_markdown() + "\n\n" + occlusion_recall.to_markdown()
    )

    print(f"run_id: {run_id}")
    print(f"trainable/total params: {trainable}/{total} ({100*trainable/total:.1f}%)")
    print(f"wall clock: {duration_sec:.1f}s")
    print(best_metrics.to_markdown())
    print(f"manifest: {run_dir / 'manifest.json'}")


def run_maskrcnn_training(config: dict, args: argparse.Namespace, device: torch.device) -> None:
    """Proposal-based instance segmentation (Mask R-CNN), built to actually
    be comparable to the literature's AP50_segm figures (TAPIS 89.85,
    LACOSTE-cited range) -- the Milestone 9 centroid/offset architecture
    plateaus at AP50_segm ~0.38, a structural gap (coarse 96x96 instance
    separation) that no amount of tuning within that architecture family
    closes. Reuses `fit_detection`/`train_one_epoch_detection` unchanged:
    Mask R-CNN's train-mode forward returns a loss dict with the same shape
    (box/RPN losses plus a new `loss_mask` term), and `sum(loss_dict.values())`
    already used there needs no change to include it. Checkpoint selection
    stays box mAP@50 (same convention, and appropriate here since we're
    warm-starting box quality that's already been separately validated).

    `model.warm_start_checkpoint` in config (optional) points at a trained
    `fasterrcnn_mobilenet_v3` checkpoint -- loaded with `strict=False` so
    the shared backbone/RPN/box-head weights transfer and only the new mask
    head (12 tensors, verified directly) starts from scratch, instead of
    learning box localization over again from an already-solved starting
    point.
    """
    train_split, val_split = splits.resolve_train_val_split(config["data"]["split"])
    augmentation = config["data"].get("augmentation", "none")

    train_ds = GraspDetectionDataset(
        args.data_root, train_split, include_masks=True,
        transform=build_detection_transforms(train=True, augmentation=augmentation),
    )
    val_ds = GraspDetectionDataset(
        args.data_root, val_split, include_masks=True,
        transform=build_detection_transforms(train=False),
    )
    class_names = train_ds.class_names_ordered()
    coco_gt_val = dataset_to_coco_gt(val_ds)

    copy_paste_cfg = config["data"].get("copy_paste", {})
    train_loader_ds = train_ds
    if copy_paste_cfg.get("enabled", False):
        train_loader_ds = CopyPasteDetectionDataset(
            train_ds, include_masks=True,
            paste_prob=copy_paste_cfg.get("paste_prob", 0.5),
            max_pastes=copy_paste_cfg.get("max_pastes", 2),
            rare_classes=copy_paste_cfg.get("rare_classes"),
            occlusion_bias=copy_paste_cfg.get("occlusion_bias", 0.7),
            seed=config["training"]["seed"],
        )

    train_loader = torch.utils.data.DataLoader(
        train_loader_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = build_detector(
        config["model"]["name"], num_classes=len(class_names),
        pretrained=config["model"]["pretrained"],
    ).to(device)

    warm_start_path = config["model"].get("warm_start_checkpoint")
    if warm_start_path:
        source_state = torch.load(warm_start_path, map_location=device)
        missing, unexpected = model.load_state_dict(source_state, strict=False)
        non_mask_missing = [k for k in missing if "mask" not in k]
        if non_mask_missing or unexpected:
            raise RuntimeError(
                f"warm start from {warm_start_path} left non-mask keys unresolved -- "
                f"unexpected: {unexpected}, missing (non-mask): {non_mask_missing}"
            )
        logging.info("warm-started %d tensors from %s (%d mask-head tensors left untrained)",
                     len(source_state), warm_start_path, len(missing))

    trainable, total = count_parameters(model)
    optimizer = build_detection_optimizer(model, config["training"])

    run_id = f"{args.config.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.experiments_dir / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"

    start = time.time()
    history, best_metrics = fit_detection(
        model, train_loader, val_loader, optimizer, class_names, device,
        epochs=config["training"]["epochs"], checkpoint_path=checkpoint_path,
        coco_gt_val=coco_gt_val, patience=config["training"].get("patience"),
    )
    duration_sec = time.time() - start

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    viz.plot_curve(history.train_loss, "train loss", "Train loss", figures_dir / "loss_curve.png")
    viz.plot_training_curves(
        history.val_map50, history.val_map50_95, "mAP", "Val mAP@50 (box) vs. mAP@50:95",
        figures_dir / "map_curve.png",
    )

    # instance-level metrics at native resolution -- Mask R-CNN's masks are
    # already at the raw image's own size (GeneralizedRCNNTransform resizes
    # predictions back before returning them), unlike the centroid/offset
    # segmenter's fixed 96x96 grid, so GT masks are decoded at native
    # resolution too rather than reusing segmentation.py's downsampled
    # convention -- comparing at each architecture's own natural output
    # resolution, not forcing both through an arbitrary shared grid.
    all_pred_instances: list[list] = []
    all_gt_instances: list[list] = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for t, out in zip(targets, outputs):
                pred_masks = (out["masks"].cpu().numpy()[:, 0] >= 0.5)
                pred_labels = out["labels"].cpu().numpy() - 1  # back to 0-indexed
                pred_scores = out["scores"].cpu().numpy()
                all_pred_instances.append(list(zip(pred_masks, pred_labels.tolist(), pred_scores.tolist())))

                image_idx = int(t["image_id"].item())
                _file_name, anns = val_ds.samples[image_idx]
                gts = []
                for a in anns:
                    gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
                    if not gt_mask.any():
                        continue
                    gts.append((gt_mask, val_ds._id_to_index[a["category_id"]]))
                all_gt_instances.append(gts)

    ap50_segm = evaluate_instance_ap50(all_pred_instances, all_gt_instances, class_names)

    occlusion_fractions = compute_occlusion_fractions(val_ds)
    counts = {"isolated": 0, "light": 0, "heavy": 0}
    hits = {"isolated": 0, "light": 0, "heavy": 0}
    for image_idx in range(len(val_ds.samples)):
        _file_name, anns = val_ds.samples[image_idx]
        preds = [(m, lbl, s) for m, lbl, s in all_pred_instances[image_idx] if s >= 0.3]
        for a in anns:
            gt_mask = decode_instance_mask(a["segmentation"]).astype(bool)
            if not gt_mask.any():
                continue
            label = val_ds._id_to_index[a["category_id"]]
            frac = occlusion_fractions.get(a["id"], 0.0)
            bucket = "isolated" if frac <= 0.0 else ("heavy" if frac > 0.5 else "light")
            counts[bucket] += 1
            matched = any(lbl == label and mask_iou(gt_mask, m) >= 0.5 for m, lbl, _s in preds)
            if matched:
                hits[bucket] += 1
    occlusion_recall_table = "\n".join(
        f"{b}: n={counts[b]} recall={hits[b] / counts[b] if counts[b] else float('nan'):.3f}"
        for b in ("isolated", "light", "heavy")
    )
    print(occlusion_recall_table)
    print(f"AP50_segm: {ap50_segm['map50']:.4f}")

    split_paths = {
        train_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[train_split],
        val_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[val_split],
    }
    manifest = {
        "run_id": run_id,
        "config": config,
        "config_path": str(args.config),
        "git": git_info(),
        "seed": config["training"]["seed"],
        "split_files": {
            name: {"path": str(path), "sha256": sha256_of(path)}
            for name, path in split_paths.items()
        },
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
        "gpu": gpu_info(device),
        "trainable_params": trainable,
        "total_params": total,
        "wall_clock_seconds": duration_sec,
        "best_checkpoint": str(checkpoint_path),
        "final_metrics": {
            "map50_box": best_metrics.map50,
            "map50_95_box": best_metrics.map50_95,
            "per_class_ap50_box": best_metrics.per_class_ap50,
            "ap50_segm": ap50_segm["map50"],
            "per_class_ap50_segm": ap50_segm["per_class_ap50"],
            "occlusion_stratified_recall_segm": {
                "n_isolated": counts["isolated"], "n_light": counts["light"], "n_heavy": counts["heavy"],
                "recall_isolated": hits["isolated"] / counts["isolated"] if counts["isolated"] else None,
                "recall_light": hits["light"] / counts["light"] if counts["light"] else None,
                "recall_heavy": hits["heavy"] / counts["heavy"] if counts["heavy"] else None,
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "metrics_table.md").write_text(
        best_metrics.to_markdown() + f"\n\nAP50_segm: {ap50_segm['map50']:.4f}\n\n" + occlusion_recall_table
    )

    print(f"run_id: {run_id}")
    print(f"trainable/total params: {trainable}/{total} ({100*trainable/total:.1f}%)")
    print(f"wall clock: {duration_sec:.1f}s")
    print(best_metrics.to_markdown())
    print(f"manifest: {run_dir / 'manifest.json'}")


def run_segmentation_training(config: dict, args: argparse.Namespace, device: torch.device) -> None:
    """Milestone 9's centroid/offset segmenter. Self-contained for the same
    reason `run_detection_training` is: a different model interface (dict
    of three per-pixel heads, no external loss_fn for the classification
    tasks' shared code below) and different checkpoint-selection metric
    (val mIoU, not macro-F1 or mAP).
    """
    train_split, val_split = splits.resolve_train_val_split(config["data"]["split"])
    image_size = config["data"].get("image_size", 384)
    output_stride = config["data"].get("output_stride", 4)

    train_ds = GraspSegmentationDataset(args.data_root, train_split, image_size=image_size, output_stride=output_stride)
    val_ds = GraspSegmentationDataset(args.data_root, val_split, image_size=image_size, output_stride=output_stride)
    class_names = train_ds.class_names_ordered()

    copy_paste_cfg = config["data"].get("copy_paste", {})
    train_loader_ds = train_ds
    if copy_paste_cfg.get("enabled", False):
        train_loader_ds = SegmentationCopyPasteDataset(
            train_ds,
            paste_prob=copy_paste_cfg.get("paste_prob", 0.5),
            max_pastes=copy_paste_cfg.get("max_pastes", 2),
            rare_classes=copy_paste_cfg.get("rare_classes"),
            occlusion_bias=copy_paste_cfg.get("occlusion_bias", 0.7),
            seed=config["training"]["seed"],
        )

    train_loader = torch.utils.data.DataLoader(
        train_loader_ds, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=args.num_workers, collate_fn=segmentation_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=args.num_workers, collate_fn=segmentation_collate_fn,
    )

    model = build_segmenter(
        config["model"]["name"], num_classes=len(class_names), pretrained=config["model"]["pretrained"],
    ).to(device)
    trainable, total = count_parameters(model)
    optimizer = build_detection_optimizer(model, config["training"])

    run_id = f"{args.config.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.experiments_dir / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"

    loss_cfg = config.get("loss", {})
    scheduler = None
    if config["training"].get("lr_schedule") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    start = time.time()
    history, best_metrics = fit_segmentation(
        model, train_loader, val_loader, optimizer, class_names, device,
        epochs=config["training"]["epochs"], checkpoint_path=checkpoint_path,
        patience=config["training"].get("patience"),
        offset_weight=loss_cfg.get("offset_weight", 0.1),
        semantic_weight=loss_cfg.get("semantic_weight", 1.0),
        scheduler=scheduler,
    )
    duration_sec = time.time() - start

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    viz.plot_curve(history.train_loss, "train loss", "Train loss", figures_dir / "loss_curve.png")
    viz.plot_curve(history.val_miou, "val mIoU", "Val mIoU", figures_dir / "miou_curve.png")

    # instance-level metrics (AP50_segm, occlusion-stratified recall) are
    # only computed once, on the best checkpoint -- same convention
    # run_detection_training uses for its occlusion analysis, since these
    # need per-image instance decoding and aren't cheap to run every epoch.
    out_hw = image_size // output_stride
    all_pred_instances: list[list] = []
    all_gt_instances: list[list] = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            heatmaps = torch.sigmoid(outputs["heatmap"]).cpu().numpy()
            offsets = outputs["offset"].cpu().numpy()
            semantics = outputs["semantic"].argmax(dim=1).cpu().numpy()
            for i in range(images.shape[0]):
                all_pred_instances.append(decode_instances(heatmaps[i], offsets[i], semantics[i]))
                gts = []
                for m, lbl in zip(targets["instance_masks"][i].numpy(), targets["instance_labels"][i].tolist()):
                    small = downsample_mask_nearest(m.astype(bool), output_stride, out_hw, out_hw)
                    gts.append((small, lbl))
                all_gt_instances.append(gts)

    ap50 = evaluate_instance_ap50(all_pred_instances, all_gt_instances, class_names)
    occlusion_fractions = compute_occlusion_fractions(val_ds)
    occlusion_recall = evaluate_occlusion_stratified_recall_segm(
        val_ds, all_pred_instances, occlusion_fractions, output_stride, score_threshold=0.3
    )
    print(occlusion_recall.to_markdown())
    print(f"AP50_segm: {ap50['map50']:.4f}")

    split_paths = {
        train_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[train_split],
        val_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[val_split],
    }
    manifest = {
        "run_id": run_id,
        "config": config,
        "config_path": str(args.config),
        "git": git_info(),
        "seed": config["training"]["seed"],
        "split_files": {
            name: {"path": str(path), "sha256": sha256_of(path)}
            for name, path in split_paths.items()
        },
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
        "gpu": gpu_info(device),
        "trainable_params": trainable,
        "total_params": total,
        "wall_clock_seconds": duration_sec,
        "best_checkpoint": str(checkpoint_path),
        "final_metrics": {
            "miou": best_metrics.miou,
            "mean_dice": best_metrics.mean_dice,
            "per_class_iou": best_metrics.per_class_iou,
            "per_class_dice": best_metrics.per_class_dice,
            "ap50_segm": ap50["map50"],
            "per_class_ap50_segm": ap50["per_class_ap50"],
            "occlusion_stratified_recall": {
                "n_isolated": occlusion_recall.n_isolated,
                "n_light": occlusion_recall.n_light,
                "n_heavy": occlusion_recall.n_heavy,
                "recall_isolated": occlusion_recall.recall_isolated,
                "recall_light": occlusion_recall.recall_light,
                "recall_heavy": occlusion_recall.recall_heavy,
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "metrics_table.md").write_text(
        best_metrics.to_markdown() + "\n\n" + occlusion_recall.to_markdown() + f"\n\nAP50_segm: {ap50['map50']:.4f}"
    )

    print(f"run_id: {run_id}")
    print(f"trainable/total params: {trainable}/{total} ({100*trainable/total:.1f}%)")
    print(f"wall clock: {duration_sec:.1f}s")
    print(best_metrics.to_markdown())
    print(f"manifest: {run_dir / 'manifest.json'}")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    set_seed(config["training"]["seed"])
    device = torch.device(args.device)

    task = config["task"]
    if task == "detection":
        run_detection_training(config, args, device)
        return
    elif task == "segmentation":
        run_segmentation_training(config, args, device)
        return
    elif task == "instance_segmentation":
        run_maskrcnn_training(config, args, device)
        return
    elif task == "multilabel_frame":
        train_loader, val_loader, class_names, loss_fn, evaluate_fn = _setup_multilabel_task(
            config, args, device
        )
    elif task == "region_classification":
        train_loader, val_loader, class_names, loss_fn, evaluate_fn = _setup_region_task(
            config, args, device
        )
    else:
        raise ValueError(
            f"unknown task '{task}'. Valid: multilabel_frame, region_classification, "
            "detection, segmentation, instance_segmentation"
        )

    model = build_model(
        config["model"]["name"], num_classes=len(class_names),
        pretrained=config["model"]["pretrained"], freeze_backbone=config["model"]["freeze_backbone"],
    )
    trainable, total = count_parameters(model)

    optimizer = build_optimizer(model, config["training"])

    run_id = f"{args.config.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = args.experiments_dir / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"

    start = time.time()
    history, best_metrics = fit(
        model, train_loader, val_loader, loss_fn, optimizer, class_names, device,
        epochs=config["training"]["epochs"], checkpoint_path=checkpoint_path,
        evaluate_fn=evaluate_fn,
    )
    duration_sec = time.time() - start

    # Re-evaluate the best checkpoint (fit() keeps history from every epoch,
    # but best_metrics is already the metrics at the best epoch).
    viz.plot_training_curves(
        history.train_loss, history.val_loss, f"{config['loss']['type']} loss", "Train/val loss",
        figures_dir / "loss_curve.png",
    )
    viz.plot_training_curves(
        history.train_macro_f1, history.val_macro_f1, "macro F1", "Train/val macro-F1",
        figures_dir / "f1_curve.png",
    )
    viz.plot_prf_bars(
        best_metrics.per_class_precision, best_metrics.per_class_recall,
        best_metrics.per_class_f1, figures_dir / "per_class_prf.png",
    )

    if task == "multilabel_frame":
        viz.plot_confusion_matrices(
            best_metrics.confusion_matrices, figures_dir / "confusion_matrices.png"
        )
        final_metrics = {
            "mean_ap": best_metrics.mean_ap,
            "macro_f1": best_metrics.macro_f1,
            "per_class_ap": best_metrics.per_class_ap,
            "per_class_f1": best_metrics.per_class_f1,
            "per_class_precision": best_metrics.per_class_precision,
            "per_class_recall": best_metrics.per_class_recall,
        }
    else:
        viz.plot_multiclass_confusion_matrix(
            best_metrics.confusion_matrix, class_names, figures_dir / "confusion_matrix.png"
        )
        final_metrics = {
            "accuracy": best_metrics.accuracy,
            "macro_f1": best_metrics.macro_f1,
            "per_class_f1": best_metrics.per_class_f1,
            "per_class_precision": best_metrics.per_class_precision,
            "per_class_recall": best_metrics.per_class_recall,
        }

    train_split, val_split = splits.resolve_train_val_split(config["data"]["split"])
    split_paths = {
        train_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[train_split],
        val_split: splits.annotations_dir(args.data_root) / splits.SHORT_TERM_SPLITS[val_split],
    }
    manifest = {
        "run_id": run_id,
        "config": config,
        "config_path": str(args.config),
        "git": git_info(),
        "seed": config["training"]["seed"],
        "split_files": {
            name: {"path": str(path), "sha256": sha256_of(path)}
            for name, path in split_paths.items()
        },
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
        },
        "gpu": gpu_info(device),
        "trainable_params": trainable,
        "total_params": total,
        "wall_clock_seconds": duration_sec,
        "best_checkpoint": str(checkpoint_path),
        "final_metrics": final_metrics,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (run_dir / "metrics_table.md").write_text(best_metrics.to_markdown())

    print(f"run_id: {run_id}")
    print(f"trainable/total params: {trainable}/{total} ({100*trainable/total:.1f}%)")
    print(f"wall clock: {duration_sec:.1f}s")
    print(best_metrics.to_markdown())
    print(f"manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
