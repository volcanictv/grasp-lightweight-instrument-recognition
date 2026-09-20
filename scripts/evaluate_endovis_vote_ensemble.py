"""Zero-shot generalization of the vote-based ensemble on EndoVis 2017/2018
(docs/DECISIONS.md 2026-09-20). Same instance extraction and crop recipe as
evaluate_endovis2018_generalization.py / evaluate_endovis2017_generalization.py
(imported, not copied); only the prediction differs: each of the 4 retrained
members makes MC-dropout passes on the crop, every pass votes for one class, and
the class with the most weighted votes wins (ties: dropout-off votes, then lower
class index). No softmax, no retraining, no tracking.

Note on comparability: the earlier EndoVis 2017 number was a single ResNet-50; this
runs the full ensemble on both datasets, as the 2018 number always did.

Usage:
    python scripts/evaluate_endovis_vote_ensemble.py --dataset 2018 --zip ~/Desktop/endovis_data/endovis2018.zip
    python scripts/evaluate_endovis_vote_ensemble.py --dataset 2017 --zip ~/Desktop/endovis_data/endovis2017.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import evaluate_endovis2017_generalization as e17
import evaluate_endovis2018_generalization as e18
from analyze_uncertainty_signals import enable_mc_dropout
from build_tracking_sample_b import plurality
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

CLASS_NAMES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
               "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def member_votes(model, crops: list[np.ndarray], image_size: int, samples: int, device: torch.device, batch: int):
    """Returns (mc_vote_share (N, C), det_vote_share (N, C)) for one member."""
    transform = build_transforms(image_size, train=False)
    mc_votes, det_votes = [], []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.stack([transform(Image.fromarray(c)) for c in crops[i:i + batch]]).to(device)
            model.eval()
            det = model(x).argmax(dim=1).cpu().numpy()
            enable_mc_dropout(model)
            logits = model(x.repeat_interleave(samples, dim=0)).view(len(x), samples, -1)
            model.eval()
            mc_votes.append((logits.argmax(dim=2)[..., None].cpu().numpy() == np.arange(7)).mean(axis=1))
            det_votes.append(np.eye(7)[det])
    return np.concatenate(mc_votes), np.concatenate(det_votes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["2017", "2018"], required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    zf = zipfile.ZipFile(args.zip)
    if args.dataset == "2018":
        instances = [(img, m, b, name) for img, m, b, name, _src in e18.extract_instances(zf, "val")]
        n_shared = 6
    else:
        instances = e17.extract_instances(zf)
        n_shared = 4
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    y = np.array([name_to_idx[name] for *_x, name in instances])
    print(f"EndoVis {args.dataset}: {len(instances)} instances", dict(Counter(name for *_x, name in instances)))

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    w320 = cfg["weight_resnet50_320"]
    members = cfg["members"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    v_mc = np.zeros((len(y), 7))
    v_det = np.zeros((len(y), 7))
    for w, m in zip(weights, members):
        crops = [e18.crop_instance(img, mask, bbox, m["letterbox"]) for img, mask, bbox, _n in instances]
        model = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        batch = 4 if m["image_size"] >= 320 else 8
        mc, det = member_votes(model, crops, m["image_size"], args.mc_samples, device, batch)
        v_mc += w * mc
        v_det += w * det
        print(f"  {m['label']}: member vote accuracy (dropout off) {(det.argmax(1) == y).mean():.4f}", flush=True)
        del model
    pred = plurality(v_mc, v_det)

    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=list(range(7)), average=None, zero_division=0)
    present = [i for i in range(n_shared) if support[i] > 0]
    result = {
        "dataset": args.dataset, "n_instances": int(len(y)), "shared_classes": n_shared, "classes_present": len(present),
        "accuracy": float((pred == y).mean()), "macro_f1_present": float(f1_score(y, pred, average="macro", labels=present, zero_division=0)),
        "per_class": {n: {"support": int(support[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                      for i, n in enumerate(CLASS_NAMES)},
        "confusion_counts": confusion_matrix(y, pred, labels=list(range(7))).tolist(), "mc_samples": args.mc_samples,
    }
    print(f"\naccuracy {result['accuracy']:.4f}, macro-F1 ({len(present)} classes present) {result['macro_f1_present']:.4f}")
    for n, v in result["per_class"].items():
        if v["support"]:
            print(f"  {n:<28} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} support={v['support']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
