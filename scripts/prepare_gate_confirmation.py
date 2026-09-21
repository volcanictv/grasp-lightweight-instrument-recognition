"""Glue for the gate confirmation (docs/DECISIONS.md 2026-09-20): writes the
fold1 ensemble config from the freshly trained members, and after the logits
cache exists, the two shards of instances to track (every u >= 0.03).

Usage:
    python scripts/prepare_gate_confirmation.py config
    python scripts/prepare_gate_confirmation.py sets --out-dir experiments/gate_confirm_fold1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import yaml

from build_tracking_sample_b import vote_share

MEMBERS = [
    ("resnet50_320", "region_letterbox_resnet50_320_deepdropout_fold1", "resnet50_deepdropout", 320, True),
    ("resnet50_224", "region_letterbox_resnet50_deepdropout_fold1", "resnet50_deepdropout", 224, True),
    ("baseline", "region_baseline_deepdropout_fold1", "mobilenet_v3_small_deepdropout", 224, False),
    ("letterbox_crop", "region_letterbox_crop_deepdropout_fold1", "mobilenet_v3_small_deepdropout", 224, True),
]
CONFIG = REPO_ROOT / "configs" / "region_ensemble_deepdropout_fold1.yaml"
CACHE = REPO_ROOT / "experiments" / "mc_logits_cache_fold1.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["config", "sets"])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "gate_confirm_fold1")
    ap.add_argument("--floor", type=float, default=0.03)
    args = ap.parse_args()
    if args.what == "config":
        members = []
        for label, run, model, size, letterbox in MEMBERS:
            found = sorted((REPO_ROOT / "experiments").glob(f"{run}_2*"))
            assert found, f"no run for {run}"
            members.append({"checkpoint": str(found[-1].relative_to(REPO_ROOT) / "best.pt"), "model": model,
                            "image_size": size, "letterbox": letterbox, "label": label})
        CONFIG.write_text("# Members trained on the fold2 cases (data.split: fold1), for the gate confirmation\n"
                          "# (docs/DECISIONS.md 2026-09-20). Same weighting as the shipped ensemble.\n"
                          + yaml.safe_dump({"weight_resnet50_320": 0.40, "members": members}, sort_keys=False))
        print(f"wrote {CONFIG}")
        return
    cfg = yaml.safe_load(CONFIG.read_text())
    members, w320 = cfg["members"], cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(CACHE)
    n_classes = cache["det_" + members[0]["label"]].shape[1]
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    u = np.round(1 - v_mc.max(axis=1), 9)
    need = np.where(u >= args.floor)[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for shard in (0, 1):
        (args.out_dir / f"track_indices_shard{shard}.json").write_text(
            json.dumps({"errors": [{"index": int(i)} for i in need[shard::2]]}))
    print(f"{len(need)} of {len(u)} instances to track ({len(need[0::2])} / {len(need[1::2])} per shard), "
          f"~{len(need) * 24 / 3600 / 2:.1f} h on two GPUs")


if __name__ == "__main__":
    main()
