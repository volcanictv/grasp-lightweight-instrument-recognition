"""One-off lookup for specific frames flagged during error-log review as
"no model or human can tell what that is" (2026-09-04) -- pulls each
matching test-set instance's true label, mask area fraction, and the
current weighted ensemble's prediction + confidence, so the actual
failure mode (not just "small mask") can be characterized before
proposing a targeted filter.

Usage:
    python scripts/inspect_hard_cases.py CASE041/04830.jpg CASE053/08575.jpg CASE041/05145.jpg
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml

from surgical_ai.data.mask_utils import decode_instance_mask, is_likely_occluded
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    target_files = set(sys.argv[1:])
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensemble_config = yaml.safe_load((REPO_ROOT / "configs" / "region_ensemble.yaml").read_text())
    members = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]

    class_names = None
    all_probs = []
    target_indices = None
    for member in members:
        ds = GraspRegionDataset(
            data_root, "test", transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            target_indices = [i for i, inst in enumerate(ds.instances) if inst[0] in target_files]
            if not target_indices:
                print(f"none of {target_files} found as annotated instances in the test split")
                return

        model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
        model.eval()

        probs = np.zeros((len(target_indices), len(class_names)))
        with torch.no_grad():
            for j, idx in enumerate(target_indices):
                image, _label = ds[idx]
                logits = model(image.unsqueeze(0).to(device))
                probs[j] = torch.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

        if member is members[0]:
            base_ds = ds

    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])

    for j, idx in enumerate(target_indices):
        file_name, segmentation, box, label_idx = base_ds.instances[idx]
        mask = decode_instance_mask(segmentation)
        h, w = segmentation["size"]
        area_pct = mask.sum() / (h * w) * 100
        occluded = is_likely_occluded(mask)
        probs = avg[j]
        pred_idx = int(probs.argmax())
        top2 = np.argsort(probs)[::-1][:2]

        print(f"\n{file_name}  bbox={box}")
        print(f"  true class: {class_names[label_idx]}")
        print(f"  mask area: {area_pct:.2f}% of frame, fragmented/occluded (heuristic): {occluded}")
        print(f"  ensemble prediction: {class_names[pred_idx]} (correct={pred_idx == label_idx})")
        print(f"  top-2 probs: {class_names[top2[0]]}={probs[top2[0]]:.3f}, {class_names[top2[1]]}={probs[top2[1]]:.3f}")
        print(f"  full distribution: " + ", ".join(f"{class_names[k]}={probs[k]:.3f}" for k in range(len(class_names))))


if __name__ == "__main__":
    main()
