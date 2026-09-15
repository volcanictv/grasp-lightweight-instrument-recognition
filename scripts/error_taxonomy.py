"""Root-cause breakdown of every current official-test misclassification
(2026-09-04): after two abstention gates (class-conditional area rule,
confidence threshold), what actually remains wrong, and why -- rather
than treating "93.4% accuracy" as one undifferentiated number.

Each wrong instance is checked against, in order:
1. would the confidence gate (t=0.40) abstain on it?
2. would the area gate (4%, four sensitive classes) abstain on it?
3. is it the already-documented Grasper/Suction closed-jaw pair (either
   direction)?
4. is it "unanimous wrong" -- all 4 ensemble members agree on the same
   wrong class? (the hard-core, no-available-signal failure mode found
   in the CASE041/05145.jpg review)
5. everything else ("residual, unexplained").

Buckets 1-2 are already handled by the adopted gates (the instance would
no longer produce a wrong answer, just no answer). Buckets 3-5 are what
actually remains as a wrong classification under the current protocol,
broken out by confusion pair, so root-cause work targets what's actually
left rather than the whole error set.

Usage:
    python scripts/error_taxonomy.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml

from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

AREA_SENSITIVE_CLASSES = {"Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors"}
AREA_THRESHOLD = 0.04
CONFIDENCE_THRESHOLD = 0.40
GRASPER_SUCTION = {"Laparoscopic Grasper", "Suction Instrument"}


def main() -> None:
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble_config = yaml.safe_load((REPO_ROOT / "configs" / "region_ensemble.yaml").read_text())
    members = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]

    class_names = None
    y_true = None
    area_frac = None
    all_probs = []
    all_member_preds = []
    base_instances = None
    for member in members:
        ds = GraspRegionDataset(
            data_root, "test", transform=build_transforms(member["image_size"], train=False),
            letterbox=member["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            area_frac = np.array([
                decode_instance_mask(seg).sum() / (seg["size"][0] * seg["size"][1])
                for _fn, seg, _box, _lbl in ds.instances
            ])
            base_instances = ds.instances

        model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs)
        all_probs.append(probs)
        all_member_preds.append(probs.argmax(axis=1))

    n_rest = len(members) - 1
    w_rest = (1 - weight_320) / n_rest
    avg = weight_320 * all_probs[0] + w_rest * sum(all_probs[1:])
    y_pred = avg.argmax(axis=1)
    max_conf = avg.max(axis=1)
    member_preds = np.stack(all_member_preds, axis=1)

    wrong = np.nonzero(y_pred != y_true)[0]
    print(f"total wrong (unfiltered baseline): {len(wrong)} / {len(y_true)} ({len(wrong)/len(y_true)*100:.1f}%)\n")

    caught_by_confidence, caught_by_area, remaining = [], [], []
    for i in wrong:
        if max_conf[i] < CONFIDENCE_THRESHOLD:
            caught_by_confidence.append(i)
        elif class_names[y_true[i]] in AREA_SENSITIVE_CLASSES and area_frac[i] < AREA_THRESHOLD:
            caught_by_area.append(i)
        else:
            remaining.append(i)

    print(f"caught by confidence gate (t<{CONFIDENCE_THRESHOLD}): {len(caught_by_confidence)}")
    print(f"caught by area gate (4 classes, <{AREA_THRESHOLD*100:.0f}%), not already caught above: {len(caught_by_area)}")
    print(f"remaining wrong under BOTH gates (this is what's actually left): {len(remaining)}\n")

    grasper_suction_ct, unanimous_wrong_ct, residual_ct = 0, 0, 0
    residual_pairs = Counter()
    grasper_suction_pairs = Counter()
    unanimous_pairs = Counter()

    for i in remaining:
        true_name, pred_name = class_names[y_true[i]], class_names[y_pred[i]]
        is_grasper_suction = true_name in GRASPER_SUCTION and pred_name in GRASPER_SUCTION
        is_unanimous = (member_preds[i] == y_pred[i]).all()

        if is_grasper_suction:
            grasper_suction_ct += 1
            grasper_suction_pairs[(true_name, pred_name)] += 1
        elif is_unanimous:
            unanimous_wrong_ct += 1
            unanimous_pairs[(true_name, pred_name)] += 1
        else:
            residual_ct += 1
            residual_pairs[(true_name, pred_name)] += 1

    print(f"of the {len(remaining)} remaining:")
    print(f"  Grasper/Suction closed-jaw ambiguity (documented ceiling): {grasper_suction_ct}")
    for pair, ct in grasper_suction_pairs.most_common():
        print(f"    {pair[0]} -> {pair[1]}: {ct}")
    print(f"  unanimous wrong (all 4 members agree, no signal available): {unanimous_wrong_ct}")
    for pair, ct in unanimous_pairs.most_common():
        print(f"    {pair[0]} -> {pair[1]}: {ct}")
    print(f"  residual (members split, not grasper/suction -- genuine open question): {residual_ct}")
    for pair, ct in residual_pairs.most_common():
        print(f"    {pair[0]} -> {pair[1]}: {ct}")

    result = {
        "total_wrong": int(len(wrong)),
        "caught_by_confidence": int(len(caught_by_confidence)),
        "caught_by_area": int(len(caught_by_area)),
        "remaining": int(len(remaining)),
        "grasper_suction": grasper_suction_ct,
        "grasper_suction_pairs": {f"{k[0]} -> {k[1]}": v for k, v in grasper_suction_pairs.items()},
        "unanimous_wrong": unanimous_wrong_ct,
        "unanimous_wrong_pairs": {f"{k[0]} -> {k[1]}": v for k, v in unanimous_pairs.items()},
        "residual": residual_ct,
        "residual_pairs": {f"{k[0]} -> {k[1]}": v for k, v in residual_pairs.items()},
        "residual_instances": [
            {
                "file": base_instances[i][0], "box": list(base_instances[i][2]),
                "true": class_names[y_true[i]], "pred": class_names[y_pred[i]],
                "confidence": float(max_conf[i]), "area_pct": float(area_frac[i] * 100),
            }
            for i in remaining if class_names[y_true[i]] not in GRASPER_SUCTION or class_names[y_pred[i]] not in GRASPER_SUCTION
        ],
    }
    (REPO_ROOT / "docs" / "error_taxonomy.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote docs/error_taxonomy.json")


if __name__ == "__main__":
    main()
