"""Accuracy of a three-member evidential ensemble on tracker-style crops of HELD-OUT instruments: the secondary
metric of docs/reports/training_experiments_preregistration.md. The crops are the EdgeTAM neighbour crops of the
held-out fold (scripts/build_temporal_neighbors.py), so this measures robustness to tracker masks directly, with no gate.

Usage (titanxp, repo root):
    python scripts/eval_on_neighbour_crops.py --ensemble-config configs/arms/ens_P_fold1_s42.yaml \\
        --neighbour-dir experiments/temporal_neighbors/fold1 --json-split fold1 --out result.json
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
from sklearn.metrics import f1_score

from surgical_ai.data.region_dataset import GraspRegionDataset, _pad_to_square, instance_key, load_neighbour_meta
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.evidential import alpha_from_logits
from surgical_ai.models import build_model

WEIGHTS = {"resnet50_320": 0.40, "baseline": 0.30, "letterbox_crop": 0.30}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-config", type=Path, required=True)
    ap.add_argument("--neighbour-dir", type=Path, required=True)
    ap.add_argument("--json-split", required=True, choices=["fold1", "fold2"])
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)

    ds = GraspRegionDataset(args.data_root, args.json_split, letterbox=True)
    label_of = {instance_key(fn, box): label for fn, _seg, box, label in ds.instances}
    items = [(path, label_of[key], info["offset"]) for key, opts in load_neighbour_meta(args.neighbour_dir).items() for path, info in opts]
    y = np.array([label for _p, label, _o in items])
    offsets = np.array([o for _p, _l, o in items])

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    alpha = np.zeros((len(items), 7))
    for m in cfg["members"]:
        if m["label"] not in WEIGHTS:
            continue
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        net.eval()
        transform = build_transforms(m["image_size"], train=False)
        logits = []
        for start in range(0, len(items), args.batch):
            batch = []
            for path, _l, _o in items[start:start + args.batch]:
                crop = np.array(Image.open(path).convert("RGB"))
                batch.append(transform(Image.fromarray(_pad_to_square(crop) if m["letterbox"] else crop)))
            with torch.no_grad():
                logits.append(net(torch.stack(batch).to(device)).float().cpu().numpy())
        alpha += WEIGHTS[m["label"]] * alpha_from_logits(np.concatenate(logits))
    pred = alpha.argmax(axis=1)
    result = {"n_crops": int(len(y)), "accuracy": float((pred == y).mean()),
              "macro_f1": float(f1_score(y, pred, average="macro", labels=range(7))),
              "per_class_f1": f1_score(y, pred, average=None, labels=range(7)).round(4).tolist(),
              "accuracy_by_abs_offset": {str(o): float((pred[np.abs(offsets) == o] == y[np.abs(offsets) == o]).mean()) for o in sorted(set(np.abs(offsets).tolist()))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("n_crops", "accuracy", "macro_f1")}))


if __name__ == "__main__":
    main()
