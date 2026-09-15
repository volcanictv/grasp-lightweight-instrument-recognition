"""Per-member vote breakdown for one specific instance, to check whether
ensemble disagreement (not just the combined confidence) would flag the
confidently-wrong Bipolar Forceps / Large Needle Driver case found during
2026-09-04 hard-case review (CASE041/05145.jpg, the 683x400 box) --
weight_resnet50_320=0.40 means that one strong, wrong member can drag the
weighted average's confidence up even if other members disagree.

Usage:
    python scripts/inspect_hard_case_members.py CASE041/05145.jpg 0 196 683 400
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import yaml

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    target_file = sys.argv[1]
    target_box = tuple(int(v) for v in sys.argv[2:6])
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensemble_config = yaml.safe_load((REPO_ROOT / "configs" / "region_ensemble.yaml").read_text())
    members = ensemble_config["members"]

    class_names = None
    for member in members:
        ds = GraspRegionDataset(
            data_root, "test", transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            target_idx = next(
                i for i, inst in enumerate(ds.instances)
                if inst[0] == target_file and inst[2] == target_box
            )

        model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
        model.eval()

        image, label = ds[target_idx]
        with torch.no_grad():
            probs = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        print(f"{member['label']:<22} (weight={'0.40' if member['label']=='resnet50_320' else '0.20'}): "
              f"predicts {class_names[pred_idx]} (p={probs[pred_idx]:.3f}), "
              f"true class prob={probs[class_names.index('Bipolar Forceps')]:.3f}")


if __name__ == "__main__":
    main()
