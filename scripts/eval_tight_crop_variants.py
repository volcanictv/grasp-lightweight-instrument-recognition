"""Scores trained tight-crop-experiment checkpoints on three views of the
validation split (docs/reports/tight_crop/):

  standard    the project's standard evaluation, unchanged (mask-only letterbox
              crops); must reproduce the manifest's best-epoch metrics
  raw_bbox    untouched original pixels inside the annotated bbox, no mask and
              no buffer (other instruments and tissue stay visible): the
              "original test frames with no alteration" robustness check
  tight_band  the training-style instrument + band crops, as a matched-
              distribution reference

Usage:
    python scripts/eval_tight_crop_variants.py --experiments DIR --out FILE [--device cuda:0]
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

from surgical_ai.data import splits
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.tight_crop import RawBBoxCropDataset, TightCropDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.classification import evaluate_region_classification
from surgical_ai.models import build_model


@torch.no_grad()
def predict(model, dataset, device, num_workers: int) -> tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers)
    preds, labels = [], []
    for images, y in loader:
        preds.append(model(images.to(device)).argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))
    device = torch.device(args.device)

    results = {}
    for manifest_path in sorted(args.experiments.glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        cfg = m["config"]
        data = cfg["data"]
        _train_split, val_split = splits.resolve_train_val_split(data["split"])
        eval_transform = build_transforms(data["image_size"], train=False)
        views = {
            "standard": GraspRegionDataset(data_root, val_split, transform=eval_transform, letterbox=data.get("letterbox_crop", False)),
            "raw_bbox": RawBBoxCropDataset(data_root, val_split, transform=eval_transform),
            "tight_band": TightCropDataset(data_root, val_split, transform=eval_transform,
                                           band_frac=data.get("tight_band_frac", 0.5), r_cap=data.get("tight_r_cap", 24)),
        }
        class_names = views["standard"].class_names_ordered()
        model = build_model(cfg["model"]["name"], num_classes=len(class_names), pretrained=False, freeze_backbone=False)
        model.load_state_dict(torch.load(m["best_checkpoint"] if Path(m["best_checkpoint"]).exists()
                                         else manifest_path.parent / "best.pt", map_location="cpu"))
        model.to(device).eval()

        out = {"crop": data.get("crop_variant", "standard"), "aug": data.get("augmentation", "default"),
               "split": data["split"], "seed": cfg["training"]["seed"]}
        for name, ds in views.items():
            y_pred, y_true = predict(model, ds, device, args.num_workers)
            met = evaluate_region_classification(y_true, y_pred, class_names)
            out[name] = {"accuracy": met.accuracy, "macro_f1": met.macro_f1, "per_class_f1": met.per_class_f1}
        drift = abs(out["standard"]["macro_f1"] - m["final_metrics"]["macro_f1"])
        out["standard_matches_manifest"] = bool(drift < 1e-3)
        results[m["run_id"]] = out
        print(f"{m['run_id']}: standard {out['standard']['macro_f1']:.4f} (manifest {m['final_metrics']['macro_f1']:.4f}), "
              f"raw_bbox {out['raw_bbox']['macro_f1']:.4f}, tight_band {out['tight_band']['macro_f1']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
