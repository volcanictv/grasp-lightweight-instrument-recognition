"""Deterministic and MC-dropout logits of an ensemble on a held-out fold, in the
same cache layout as analyze_mc_logit_signals.py (which is hard-wired to the
official test split). Used for the gate confirmation on cases the fold-trained
members never saw (docs/DECISIONS.md 2026-09-20).

Usage:
    python scripts/build_fold_logits_cache.py --ensemble-config configs/region_ensemble_deepdropout_fold1.yaml \
        --split fold1 --data-root GraSP --out experiments/mc_logits_cache_fold1.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml

from analyze_uncertainty_signals import enable_mc_dropout
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-config", type=Path, required=True)
    ap.add_argument("--split", default="fold1")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mc-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    members = yaml.safe_load(args.ensemble_config.read_text())["members"]
    arrays, y_true = {}, None
    for m in members:
        ds = GraspRegionDataset(args.data_root, args.split, transform=build_transforms(m["image_size"], train=False),
                                letterbox=m["letterbox"])
        y = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
        assert y_true is None or (y == y_true).all(), "member datasets disagree on instance order"
        y_true = y
        model = build_model(m["model"], num_classes=len(ds.class_names_ordered()), pretrained=False,
                            freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        det, mc = [], [[] for _ in range(args.mc_samples)]
        with torch.no_grad():
            for images, _labels in loader:
                x = images.to(device)
                model.eval()
                det.append(model(x).float().cpu().numpy())
                enable_mc_dropout(model)
                for s in range(args.mc_samples):
                    mc[s].append(model(x).float().cpu().numpy())
        arrays[f"det_{m['label']}"] = np.concatenate(det)
        arrays[f"mc_{m['label']}"] = np.stack([np.concatenate(p) for p in mc])
        print(f"{m['label']}: {len(y)} instances, dropout-off accuracy {(arrays[f'det_{m['label']}'].argmax(1) == y).mean():.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, y_true=y_true, **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
