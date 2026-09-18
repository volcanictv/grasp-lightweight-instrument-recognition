"""Test-time-only probe of deeper MC Dropout (docs/DECISIONS.md 2026-09-18).

Every member currently has one Dropout layer, right before the final Linear,
so MC Dropout only perturbs the classifier head. This injects channel-wise
dropout (F.dropout2d, forward hooks) deeper into the EXISTING checkpoints at
inference time only -- no retraining -- and saves per-pass logits for each
placement/rate so the complementarity analysis can be rerun on them:

  late     : after the last conv stage (ResNet-50 layer4 / MobileNetV3 features[12])
  midlate  : late plus one mid-depth stage (ResNet-50 layer3 / MobileNetV3 features[8])

each at p in {0.1, 0.2, 0.3}, on top of the existing head dropout (kept
stochastic, p=0.2). The networks were not trained with this dropout, so the
perturbation is off-distribution; that is the known limit of this probe, and
a positive result would still need a retrained confirmation.

Two stages:
    python scripts/probe_deep_dropout.py collect --members resnet50_320 --device cuda:0
    python scripts/probe_deep_dropout.py assemble
`assemble` writes one cache per config in the format analyze_mc_logit_signals.py /
complementarity_analysis.py read (--logits-cache).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from analyze_uncertainty_signals import enable_mc_dropout
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

CONFIGS = {f"{place}_p{int(p * 100):02d}": (place, p) for place in ("late", "midlate") for p in (0.1, 0.2, 0.3)}
OUT_DIR = REPO_ROOT / "experiments" / "deep_dropout"


def hook_targets(model: torch.nn.Module, model_name: str, placement: str) -> list:
    if model_name.startswith("resnet50"):
        late, mid = [model.layer4], [model.layer3]
    elif model_name == "mobilenet_v3_small":
        late, mid = [model.features[12]], [model.features[8]]
    else:
        raise ValueError(f"no injection points defined for {model_name}")
    return late if placement == "late" else mid + late


def collect(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(args.ensemble_config.read_text())
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for m in [m for m in cfg["members"] if m["label"] in args.members]:
        ds = GraspRegionDataset(
            data_root, "test", transform=build_transforms(m["image_size"], train=False), letterbox=m["letterbox"],
        )
        class_names = ds.class_names_ordered()
        y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
        model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)

        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        det = []
        mc = {name: [[] for _ in range(args.mc_samples)] for name in CONFIGS}
        with torch.no_grad():
            for images, _labels in loader:
                x = images.to(device)
                model.eval()
                det.append(model(x).float().cpu().numpy())
                for name, (placement, p) in CONFIGS.items():
                    enable_mc_dropout(model)
                    handles = [
                        t.register_forward_hook(lambda _mod, _inp, out, p=p: F.dropout2d(out, p=p, training=True))
                        for t in hook_targets(model, m["model"], placement)
                    ]
                    for s in range(args.mc_samples):
                        mc[name][s].append(model(x).float().cpu().numpy())
                    for h in handles:
                        h.remove()
        out = {"y_true": y_true, "det": np.concatenate(det)}
        for name in CONFIGS:
            out[f"mc_{name}"] = np.stack([np.concatenate(p_) for p_ in mc[name]])
        np.savez_compressed(OUT_DIR / f"member_{m['label']}.npz", **out)
        print(f"wrote member_{m['label']}.npz", flush=True)


def assemble(args: argparse.Namespace) -> None:
    labels = [m["label"] for m in yaml.safe_load(args.ensemble_config.read_text())["members"]]
    members = {label: np.load(OUT_DIR / f"member_{label}.npz") for label in labels}
    y_true = members[labels[0]]["y_true"]
    for label in labels:
        assert (members[label]["y_true"] == y_true).all(), "member instance order differs"
    for name in CONFIGS:
        cache = {"y_true": y_true}
        for label in labels:
            cache[f"det_{label}"] = members[label]["det"]
            cache[f"mc_{label}"] = members[label][f"mc_{name}"]
        np.savez_compressed(OUT_DIR / f"cache_{name}.npz", **cache)
        print(f"wrote cache_{name}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["collect", "assemble"])
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_mcdropout.yaml")
    parser.add_argument("--members", nargs="+", default=["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    a = parser.parse_args()
    collect(a) if a.stage == "collect" else assemble(a)
