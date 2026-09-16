"""Finds the 4-way Task B ensemble's actual misclassified instances on
official test (single ground-truth frame, no propagation) -- the targeted
population for testing whether SAM2 temporal-track aggregation fixes real
errors, instead of running the expensive propagation pipeline against the
whole 2,861-instance test set (`docs/DECISIONS.md` 2026-09-15: killed
early, that run was going to take ~5.4h for a question this narrows down
to a much smaller, cheaper set).

Read-only, no SAM2, just the ensemble's own single-frame predictions
(`configs/region_ensemble.yaml` weighting) vs. ground truth.

Usage:
    python scripts/find_ensemble_error_cases.py
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

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    ensemble_config_path = REPO_ROOT / "configs" / "region_ensemble.yaml"
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split = "test"
    out_path = REPO_ROOT / "docs" / "reports" / "ensemble_official_test_errors.json"

    ensemble_config = yaml.safe_load(ensemble_config_path.read_text())
    members_cfg = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]
    w_rest = (1 - weight_320) / (len(members_cfg) - 1)

    class_names = None
    y_true = None
    area_fracs = None
    combined_probs = None
    base_instances = None

    for m in members_cfg:
        ds = GraspRegionDataset(
            data_root, split, transform=build_transforms(m["image_size"], train=False), letterbox=m["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            area_fracs = np.array([
                decode_instance_mask(seg).sum() / (seg["size"][0] * seg["size"][1])
                for _fn, seg, _box, _lbl in ds.instances
            ])
            base_instances = ds.instances
            combined_probs = np.zeros((len(ds.instances), len(class_names)), dtype=np.float64)

        model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs)

        weight = weight_320 if m["label"] == "resnet50_320" else w_rest
        combined_probs += weight * probs
        print(f"scored {m['label']} (weight={weight:.2f})")

    y_pred = combined_probs.argmax(axis=1)
    confidence = combined_probs.max(axis=1)
    acc = float((y_pred == y_true).mean())
    print(f"\nofficial test: n={len(y_true)}, ensemble single-frame accuracy={acc:.4f}")

    wrong = np.nonzero(y_pred != y_true)[0]
    print(f"misclassified: {len(wrong)} / {len(y_true)} ({len(wrong)/len(y_true)*100:.1f}%)")

    errors = [
        {
            "index": int(i),
            "file": base_instances[i][0],
            "box": list(base_instances[i][2]),
            "true": class_names[y_true[i]],
            "pred": class_names[y_pred[i]],
            "confidence": float(confidence[i]),
            "area_pct": float(area_fracs[i] * 100),
        }
        for i in wrong
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_total": int(len(y_true)), "n_errors": int(len(wrong)),
                                     "accuracy": acc, "errors": errors}, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
