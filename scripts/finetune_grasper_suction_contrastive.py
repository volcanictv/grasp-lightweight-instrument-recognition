"""Targeted fine-tune for the one confusion nothing else touched this
session: Laparoscopic Grasper vs. Suction Instrument. Root cause found by
direct visual inspection (docs/DECISIONS.md, 2026-09-03): when the
Grasper's jaws are closed, it's a smooth featureless shaft indistinguishable
from a Suction Instrument's tube -- no visual information is present to
tell them apart in that state. Capacity, resolution, ensembling, and both
narrow- and wide-window temporal search were all tried and none touched
it (docs/DECISIONS.md). This is the one remaining untried lever: a
contrastive term that explicitly pushes these two classes' embeddings
apart during fine-tuning, on the chance a faint residual signal (material
sheen, taper, subtle color) exists that plain cross-entropy isn't forcing
the model to use. If it doesn't help, that's real evidence the confusion
is a genuine ceiling, not a training-objective problem.

Warm-starts from the best existing single model (ResNet-50@320) rather
than training from scratch -- this is a fine-tune, not a new run. Keeps
the original checkpoint untouched; saves the fine-tuned result separately
so a regression doesn't cost anything already validated.

Usage:
    python scripts/finetune_grasper_suction_contrastive.py \\
        --base-checkpoint experiments/region_letterbox_resnet50_320_20260902-234246/best.pt \\
        --lambda-contrastive 1.0 --margin 0.3 --epochs 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model
from surgical_ai.training.losses import compute_class_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "region_grasper_suction_contrastive")
    parser.add_argument("--split", default="official")
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--backbone-lr", type=float, default=0.00003)
    parser.add_argument("--lambda-contrastive", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.3, help="cosine-similarity margin; pairs more similar than this are penalized")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, lr: float, backbone_lr: float) -> torch.optim.Optimizer:
    head_param_ids = {id(p) for p in model.head.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_param_ids]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
    return torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": lr},
    ])


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, y in loader:
            logits = model(images.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {device}")

    from surgical_ai.data import splits
    train_split, val_split = splits.resolve_train_val_split(args.split)
    train_ds = GraspRegionDataset(
        args.data_root, train_split, transform=build_transforms(args.image_size, train=True, augmentation="default"),
        letterbox=True,
    )
    val_ds = GraspRegionDataset(
        args.data_root, val_split, transform=build_transforms(args.image_size, train=False), letterbox=True,
    )
    class_names = train_ds.class_names_ordered()
    grasper_idx = class_names.index("Laparoscopic Grasper")
    suction_idx = class_names.index("Suction Instrument")
    print(f"Laparoscopic Grasper idx={grasper_idx}, Suction Instrument idx={suction_idx}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    label_counts = torch.zeros(len(class_names))
    for _, _, _, label_idx in train_ds.instances:
        label_counts[label_idx] += 1
    class_weight = compute_class_weights(label_counts).to(device)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)

    model = build_model("resnet50", num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(REPO_ROOT / args.base_checkpoint, map_location=device))

    features_capture: dict[str, torch.Tensor] = {}

    def hook(_module, _input, output):
        features_capture["feat"] = output.flatten(1)

    model.avgpool.register_forward_hook(hook)

    optimizer = build_optimizer(model, args.lr, args.backbone_lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "best.pt"

    y_pred_base, y_true = evaluate(model, val_loader, device)
    best_macro_f1 = f1_score(y_true, y_pred_base, average="macro", zero_division=0)
    print(f"baseline (pre-finetune) macro-F1 on {val_split}: {best_macro_f1:.4f}")
    torch.save(model.state_dict(), checkpoint_path)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_ce, total_contrastive, n_batches_with_pair = 0.0, 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            ce_loss = ce_loss_fn(logits, labels)
            features = features_capture["feat"]

            grasper_mask = labels == grasper_idx
            suction_mask = labels == suction_idx
            contrastive_loss = torch.tensor(0.0, device=device)
            if grasper_mask.any() and suction_mask.any():
                grasper_feats = F.normalize(features[grasper_mask], dim=1)
                suction_feats = F.normalize(features[suction_mask], dim=1)
                sim = grasper_feats @ suction_feats.T
                contrastive_loss = F.relu(sim - args.margin).mean()
                n_batches_with_pair += 1

            loss = ce_loss + args.lambda_contrastive * contrastive_loss
            loss.backward()
            optimizer.step()

            total_ce += ce_loss.item()
            total_contrastive += contrastive_loss.item()

        y_pred, y_true = evaluate(model, val_loader, device)
        val_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        _, _, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0)
        n_batches = len(train_loader)
        print(
            f"epoch {epoch}/{args.epochs} ce_loss={total_ce/n_batches:.4f} "
            f"contrastive_loss={total_contrastive/n_batches:.4f} (pair batches={n_batches_with_pair}/{n_batches}) "
            f"val_macro_f1={val_macro_f1:.4f} grasper_f1={f1_per_class[grasper_idx]:.4f} suction_f1={f1_per_class[suction_idx]:.4f}"
        )
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), checkpoint_path)

    duration = time.time() - start
    print(f"\nwall clock: {duration:.1f}s")
    print(f"best val macro-F1: {best_macro_f1:.4f}")
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
