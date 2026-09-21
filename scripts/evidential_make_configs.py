"""Training configs and ensemble configs for the evidential validation
(docs/DECISIONS.md 2026-09-21). Arm E = evidential members (plain architectures,
evidential loss); arm C = the cross-entropy deep-dropout members the vote gate
uses. Both use the same four members, image sizes, letterbox flags, optimiser,
epochs, lr and class weights as configs/region_ensemble_deepdropout.yaml.

Usage:
    python scripts/evidential_make_configs.py train --arm E --fold fold1 --seed 42 --lam 0.1 --anneal 10 [--members resnet50_320]
    python scripts/evidential_make_configs.py ensemble --arm E --fold fold1 --seed 42 --lam-tag 0p1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "configs" / "evidential"

MEMBERS = {
    "resnet50_320": dict(c="resnet50_deepdropout", e="resnet50", size=320, letterbox=True),
    "resnet50_224": dict(c="resnet50_deepdropout", e="resnet50", size=224, letterbox=True),
    "baseline": dict(c="mobilenet_v3_small_deepdropout", e="mobilenet_v3_small", size=224, letterbox=False),
    "letterbox_crop": dict(c="mobilenet_v3_small_deepdropout", e="mobilenet_v3_small", size=224, letterbox=True),
}
WEIGHT_320 = 0.40


def lam_tag(lam: float, anneal: int) -> str:
    return f"lam{str(round(lam, 4)).replace('.', 'p')}a{anneal}"


def stem(arm: str, member: str, fold: str, seed: int, tag: str) -> str:
    return f"edl_{arm}_{member}_{fold}_s{seed}" + (f"_{tag}" if arm == "E" else "")


def train_config(arm: str, member: str, fold: str, seed: int, lam: float, anneal: int) -> dict:
    m = MEMBERS[member]
    loss = {"type": "cross_entropy", "class_weights": True}
    if arm == "E":
        loss = {"type": "evidential", "class_weights": True, "edl_lambda": lam,
                "edl_anneal_epochs": anneal, "edl_clamp": 10.0}
    data = {"split": fold, "image_size": m["size"], "augmentation": "default"}
    if m["letterbox"]:
        data["letterbox_crop"] = True
    return {
        "task": "region_classification",
        "model": {"name": m["e" if arm == "E" else "c"], "pretrained": True, "freeze_backbone": False},
        "data": data,
        "loss": loss,
        "training": {"batch_size": 32, "epochs": 20, "lr": 0.001, "backbone_lr": 0.0001, "seed": seed},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["train", "ensemble"])
    ap.add_argument("--arm", choices=["E", "C"], required=True)
    ap.add_argument("--fold", choices=["fold1", "fold2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--anneal", type=int, default=10)
    ap.add_argument("--members", nargs="+", default=list(MEMBERS))
    ap.add_argument("--experiments-dir", type=Path, default=REPO_ROOT / "experiments")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = lam_tag(args.lam, args.anneal)
    if args.what == "train":
        for member in args.members:
            path = OUT / f"{stem(args.arm, member, args.fold, args.seed, tag)}.yaml"
            path.write_text(yaml.safe_dump(train_config(args.arm, member, args.fold, args.seed, args.lam, args.anneal),
                                           sort_keys=False))
            print(path.name)
        return
    members = []
    for label, m in MEMBERS.items():
        runs = sorted(args.experiments_dir.glob(f"{stem(args.arm, label, args.fold, args.seed, tag)}_2*"))
        assert runs, f"no finished run for {stem(args.arm, label, args.fold, args.seed, tag)}"
        members.append({"checkpoint": str(runs[-1].relative_to(REPO_ROOT) / "best.pt"),
                        "model": m["e" if args.arm == "E" else "c"], "image_size": m["size"],
                        "letterbox": m["letterbox"], "label": label})
    name = f"ens_{args.arm}_{args.fold}_s{args.seed}" + (f"_{tag}" if args.arm == "E" else "") + ".yaml"
    path = OUT / name
    path.write_text(yaml.safe_dump({"weight_resnet50_320": WEIGHT_320, "members": members}, sort_keys=False))
    print(path.name)


if __name__ == "__main__":
    main()
