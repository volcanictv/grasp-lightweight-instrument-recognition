"""Reproducibility check: seed 42 alone doesn't tell us how much of the
reported number is the recipe vs. this one random draw of weight init /
data order. Retrains each of the 4 ensemble members across >=5 seeds
(configs/*_seed{43..46}.yaml, seed 42 reuses the already-trained
checkpoints in configs/region_ensemble.yaml) and reports mean +/- SD of
the weighted ensemble's accuracy and macro-F1 across trials.

Plain classify-everything protocol -- no abstention/decline mechanism
(2026-09-04: coverage must never drop below 99%, so none is used).

A trial's 4 member checkpoints must all come from the same seed -- this
does not mix, e.g., seed 43's resnet50_320 with seed 44's mobilenet.

Usage:
    python scripts/evaluate_region_ensemble_seed_trials.py
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
from sklearn.metrics import f1_score

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

SEEDS = [42, 43, 44, 45, 46]

# member key -> (config_stem_without_seed_suffix, model name, image_size, letterbox)
# Keys must match the "label" fields in configs/region_ensemble.yaml (baseline,
# letterbox_crop -- not mobilenet_baseline/mobilenet_letterbox) since that's
# what resolves each member's seed-42 checkpoint.
MEMBER_SPECS = {
    "resnet50_320": ("region_letterbox_resnet50_320", "resnet50", 320, True),
    "resnet50_224": ("region_letterbox_resnet50", "resnet50", 224, True),
    "baseline": ("region_baseline", "mobilenet_v3_small", 224, False),
    "letterbox_crop": ("region_letterbox_crop", "mobilenet_v3_small", 224, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "seed_trial_results.json")
    return parser.parse_args()


def resolve_checkpoint(experiments_dir: Path, config_stem: str, seed: int, seed42_fallback: str | None) -> Path:
    if seed == 42 and seed42_fallback is not None:
        return REPO_ROOT / seed42_fallback
    candidates = sorted(experiments_dir.glob(f"{config_stem}_seed{seed}_*/best.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found for {config_stem} seed{seed} under {experiments_dir}")
    return candidates[-1]  # most recent timestamp if more than one


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.ensemble_config.read_text())
    weight_320 = ensemble_config["weight_resnet50_320"]
    seed42_checkpoints = {m["label"]: m["checkpoint"] for m in ensemble_config["members"]}

    class_names = None
    y_true = None
    trial_results = []

    for seed in SEEDS:
        probs_by_member = {}
        for label, (config_stem, model_name, image_size, letterbox) in MEMBER_SPECS.items():
            checkpoint = resolve_checkpoint(args.experiments_dir, config_stem, seed, seed42_checkpoints.get(label))

            ds = GraspRegionDataset(
                args.data_root, args.split, transform=build_transforms(image_size, train=False),
                letterbox=letterbox,
            )
            if class_names is None:
                class_names = ds.class_names_ordered()
                y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])

            model = build_model(model_name, num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
            model.eval()

            loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            probs = []
            with torch.no_grad():
                for images, _labels in loader:
                    logits = model(images.to(device))
                    probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            probs_by_member[label] = np.concatenate(probs)
            print(f"seed {seed}: {label} standalone accuracy={(probs_by_member[label].argmax(axis=1) == y_true).mean():.4f} ({checkpoint})")

        n_rest = 3
        w_rest = (1 - weight_320) / n_rest
        avg = (
            weight_320 * probs_by_member["resnet50_320"]
            + w_rest * probs_by_member["resnet50_224"]
            + w_rest * probs_by_member["baseline"]
            + w_rest * probs_by_member["letterbox_crop"]
        )
        y_pred = avg.argmax(axis=1)

        accuracy = float((y_pred == y_true).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        trial = {"seed": seed, "n": int(len(y_true)), "accuracy": accuracy, "macro_f1": macro_f1}
        trial_results.append(trial)
        print(f"seed {seed}: accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}\n")

    def mean_sd(key: str) -> tuple[float, float]:
        vals = np.array([t[key] for t in trial_results])
        return float(vals.mean()), float(vals.std(ddof=1))

    acc_mean, acc_sd = mean_sd("accuracy")
    f1_mean, f1_sd = mean_sd("macro_f1")
    summary = {
        "seeds": SEEDS,
        "trials": trial_results,
        "accuracy_mean": acc_mean,
        "accuracy_sd": acc_sd,
        "macro_f1_mean": f1_mean,
        "macro_f1_sd": f1_sd,
    }
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"\n{len(SEEDS)}-seed summary (classify-everything, 4-model ensemble):")
    print(f"  accuracy: {acc_mean:.4f} +/- {acc_sd:.4f}")
    print(f"  macro-F1: {f1_mean:.4f} +/- {f1_sd:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
