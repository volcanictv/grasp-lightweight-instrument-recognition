"""Builds a visual error gallery for the Task B classifier: every
misclassified official-test instance, grouped by confusion pair (true
class -> predicted class), saved as an actual crop image (what the model
was actually shown, not the raw frame) plus a JSON manifest with
per-instance metadata (source frame, predicted-class confidence,
true-class confidence). Uses the current reported configuration
(`configs/region_ensemble.yaml`, weighted ensemble) by default.

Display crops use the resnet50_320 ensemble member's own preprocessing
(letterbox=True, the highest-weighted and highest-resolution member) --
the different members technically see slightly different crops (different
image_size/letterbox settings), so there's no single "the" input crop for
an ensemble; this is the most representative single choice, not a claim
that every member saw pixel-identical input.

Usage:
    python scripts/generate_error_gallery.py --out-dir error_gallery
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from PIL import Image

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--display-member", default="resnet50_320", help="which ensemble member's crop style to save for display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.config.read_text())
    members = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]
    print(f"device: {device}, weight(resnet50_320)={weight_320}")

    class_names = None
    y_true = None
    file_names = None
    all_probs = []
    for member in members:
        ds = GraspRegionDataset(
            args.data_root, args.split, transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            file_names = [fn for fn, _seg, _box, _lbl in ds.instances]
        model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs))

    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)

    display_member = next(m for m in members if m["label"] == args.display_member)
    display_ds = GraspRegionDataset(args.data_root, args.split, transform=None, letterbox=display_member["letterbox"])

    wrong_idx = np.nonzero(y_pred != y_true)[0]
    print(f"{len(wrong_idx)} / {len(y_true)} misclassified ({(y_pred == y_true).mean():.4f} accuracy)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx in wrong_idx:
        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]
        pair_dir = args.out_dir / f"{true_name.replace(' ', '_')}_as_{pred_name.replace(' ', '_')}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        image, _label = display_ds[idx]
        image.thumbnail((256, 256))
        out_path = pair_dir / f"{idx:05d}.png"
        image.save(out_path)

        manifest.append({
            "idx": int(idx),
            "file_name": file_names[idx],
            "true_class": true_name,
            "pred_class": pred_name,
            "pred_confidence": float(avg[idx, y_pred[idx]]),
            "true_confidence": float(avg[idx, y_true[idx]]),
            "image_path": str(out_path.relative_to(args.out_dir)).replace("\\", "/"),
        })

    manifest.sort(key=lambda m: (m["true_class"], m["pred_class"], -m["pred_confidence"]))
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    from collections import Counter
    pair_counts = Counter((m["true_class"], m["pred_class"]) for m in manifest)
    print("\nconfusion pairs, most frequent first:")
    for (t, p), count in pair_counts.most_common():
        print(f"  {t} -> {p}: {count}")

    print(f"\nwrote {len(manifest)} crops + manifest.json to {args.out_dir}")


if __name__ == "__main__":
    main()
