"""Does adding resnet101 to the ensemble help, regardless of its
standalone accuracy? (2026-09-04) -- resnet101 replicated a macro-F1 gain
on fold1/fold2 but scored slightly worse than resnet50_320 standalone on
official test; ensembling doesn't require a new member to individually
beat the strongest one (the two MobileNet members already don't), so
this tests the actual question: does the 5-model combination beat the
current 4-model one under the adopted combined-gates protocol.

Flat (1/5 each) weighting, matching this project's own finding that flat
weighting is the validated-best scheme (region_ensemble.yaml's own
comment) rather than assuming a hand-picked weight.

Usage:
    python scripts/evaluate_region_ensemble_with_resnet101.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

AREA_SENSITIVE_CLASSES = {"Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors"}
AREA_THRESHOLD = 0.04
CONFIDENCE_THRESHOLD = 0.40


def main() -> None:
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_config = yaml.safe_load((REPO_ROOT / "configs" / "region_ensemble.yaml").read_text())
    members = list(ensemble_config["members"])
    members.append({
        "checkpoint": "experiments/region_letterbox_resnet101_320_20260904-043248/best.pt",
        "model": "resnet101", "image_size": 320, "letterbox": True, "label": "resnet101_320",
    })

    class_names = None
    y_true = None
    area_frac = None
    all_probs = {}
    for member in members:
        ds = GraspRegionDataset(
            data_root, "test", transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            area_frac = np.array([
                decode_instance_mask(seg).sum() / (seg["size"][0] * seg["size"][1])
                for _fn, seg, _box, _lbl in ds.instances
            ])

        model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs)
        all_probs[member["label"]] = probs
        print(f"  {member['label']}: standalone accuracy={(probs.argmax(axis=1) == y_true).mean():.4f}")

    def report(y_pred: np.ndarray, max_conf: np.ndarray, label: str) -> None:
        is_sensitive = np.isin(y_true, [class_names.index(n) for n in AREA_SENSITIVE_CLASSES])
        abstain = (is_sensitive & (area_frac < AREA_THRESHOLD)) | (max_conf < CONFIDENCE_THRESHOLD)
        covered = ~abstain
        acc_unfiltered = (y_pred == y_true).mean()
        f1_unfiltered = f1_score(y_true, y_pred, average="macro", zero_division=0)
        acc_covered = (y_pred[covered] == y_true[covered]).mean()
        f1_covered = f1_score(y_true[covered], y_pred[covered], average="macro", zero_division=0)
        print(f"\n{label}:")
        print(f"  unfiltered: accuracy={acc_unfiltered:.4f} macro_f1={f1_unfiltered:.4f}")
        print(f"  combined gates: n_covered={covered.sum()} ({covered.mean()*100:.1f}%) accuracy={acc_covered:.4f} macro_f1={f1_covered:.4f}")

    # current 4-model ensemble (weighted, as shipped) -- reference point
    weight_320 = ensemble_config["weight_resnet50_320"]
    current_members = ensemble_config["members"]
    w_rest = (1 - weight_320) / (len(current_members) - 1)
    current_avg = weight_320 * all_probs["resnet50_320"] + w_rest * sum(
        all_probs[m["label"]] for m in current_members if m["label"] != "resnet50_320"
    )
    report(current_avg.argmax(axis=1), current_avg.max(axis=1), "current 4-model ensemble (weighted, shipped)")

    # 5-model flat ensemble
    flat_5_avg = sum(all_probs.values()) / len(all_probs)
    report(flat_5_avg.argmax(axis=1), flat_5_avg.max(axis=1), "5-model flat ensemble (+ resnet101)")

    # 4-model flat ensemble (no resnet101) -- isolates whether adding resnet101 helps vs. just flattening weights
    current_flat_avg = sum(all_probs[m["label"]] for m in current_members) / len(current_members)
    report(current_flat_avg.argmax(axis=1), current_flat_avg.max(axis=1), "4-model flat ensemble (no resnet101, isolates weighting-vs-membership)")


if __name__ == "__main__":
    main()
