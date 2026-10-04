"""Writes the four-member evidential ensemble configs (resnet50_320, resnet50_224, baseline, letterbox_crop; the member set and weights of
the GT-box pipeline's CONFIGS["four"]) for the tracker-style-crop arms, from the finished official-split runs of scripts/run_training_arms.py.

Usage (titanxp, repo root): python scripts/make_armN_ensemble_configs.py --arm N --fold official --seeds 42 43 44
    -> configs/arms/ens4_N_official_s{seed}.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evidential_make_configs as mk
from run_training_arms import EXP, arm_stem

LABELS = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="N")
    ap.add_argument("--fold", default="official")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    for seed in args.seeds:
        members = []
        for label in LABELS:
            runs = [r for r in sorted(EXP.glob(f"{arm_stem(args.arm, label, args.fold, seed)}_2*")) if (r / "best.pt").exists()]
            if not runs:
                raise FileNotFoundError(f"no finished run for {arm_stem(args.arm, label, args.fold, seed)}")
            m = mk.MEMBERS[label]
            members.append({"checkpoint": str((runs[-1] / "best.pt").relative_to(REPO_ROOT)), "model": m["e"], "image_size": m["size"],
                            "letterbox": m["letterbox"], "label": label})
        path = REPO_ROOT / "configs" / "arms" / f"ens4_{args.arm}_{args.fold}_s{seed}.yaml"
        path.write_text(yaml.safe_dump({"weight_resnet50_320": mk.WEIGHT_320, "members": members}, sort_keys=False))
        print("wrote", path)


if __name__ == "__main__":
    main()
