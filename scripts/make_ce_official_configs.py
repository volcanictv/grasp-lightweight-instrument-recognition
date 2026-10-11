"""Training configs of the cross-entropy baseline members on the OFFICIAL split (arm C official): the recipe of the evidential finals (configs/arms/armN_<member>_official_s<seed>.yaml: val_split_override fold1,
fixed schedule, last epoch kept, tracker-style neighbour crops with probability 0.5, 20 epochs, same optimiser) with only the loss changed to cross-entropy (class-weighted) and the networks changed to their deep-dropout
variants (the ones MC dropout needs). The test cases are never read during training. Used for the softmax, MC-dropout and deep-ensemble baselines on the test crops.

    python scripts/make_ce_official_configs.py          # writes configs/arms/armC_<member>_official_s<seed>.yaml for the four members and seeds 42, 43, 44
"""
from __future__ import annotations

from pathlib import Path

import yaml

ARMS = Path(__file__).resolve().parents[1] / "configs" / "arms"
MEMBERS = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
DROPOUT_MODEL = {"resnet50": "resnet50_deepdropout", "mobilenet_v3_small": "mobilenet_v3_small_deepdropout"}


def main() -> None:
    for seed in (42, 43, 44):
        for m in MEMBERS:
            cfg = yaml.safe_load((ARMS / f"armN_{m}_official_s{seed}.yaml").read_text())
            cfg["model"]["name"] = DROPOUT_MODEL[cfg["model"]["name"]]
            cfg["loss"] = {"type": "cross_entropy", "class_weights": True}
            out = ARMS / f"armC_{m}_official_s{seed}.yaml"
            out.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(out.name, cfg["model"]["name"], cfg["data"].get("val_split_override"), cfg["training"].get("select_best"))


if __name__ == "__main__":
    main()
