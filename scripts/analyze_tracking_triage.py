"""Designs a triage rule for when to pay SAM2 temporal-track aggregation's
~13.5s/instance cost (`docs/DECISIONS.md` 2026-09-15 latency benchmark),
instead of running it on every instance. Ground-truth labels aren't
available at deployment time, so this has to decide *before* knowing
whether the ensemble is actually wrong -- using only signals available for
free from the ensemble's own single-frame prediction.

Confidence (max combined softmax) is the standard, cheapest selective-
classification signal (Chow's rule / softmax response), so it's the first
thing tried here, not a learned gate -- same "simple thing first" standard
this project applies everywhere (the learned per-frame aggregator, 2026-09-15
above, already failed to beat a simpler rule once; no reason to assume a
learned gate would fare better without checking the free baseline first).

Sweeps a confidence threshold and reports, per threshold: what fraction of
the test set gets flagged (cost), what fraction of the ensemble's real
errors are caught (recall), and an estimated net accuracy delta using the
already-measured fix rate (~54-61%, fold1/fold2 full-population + the
188 real official-test errors) and regression rate (~4-5%, correct
instances tracking breaks) from prior entries -- not re-derived here,
cited directly.

Usage:
    python scripts/analyze_tracking_triage.py
"""
from __future__ import annotations

import json
import os
import sys
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

# measured directly, not assumed -- docs/DECISIONS.md 2026-09-15 entries
FIX_RATE = 0.58  # average of fold1 (53.8%), fold2 (58.5%), official-test errors (61.2%)
REGRESSION_RATE = 0.043  # average of fold1 (4.87%), fold2 (3.57%)
SAM2_COST_S = 13.5  # median per-instance latency, SAM2 + 4-way ensemble classification


def main() -> None:
    ensemble_config_path = REPO_ROOT / "configs" / "region_ensemble.yaml"
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split = "test"
    out_path = REPO_ROOT / "docs" / "reports" / "tracking_triage_analysis.json"

    ensemble_config = yaml.safe_load(ensemble_config_path.read_text())
    members_cfg = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]
    w_rest = (1 - weight_320) / (len(members_cfg) - 1)

    class_names = None
    y_true = None
    area_fracs = None
    combined_probs = None
    per_member_argmax = []

    for m in members_cfg:
        ds = GraspRegionDataset(
            data_root, split, transform=build_transforms(m["image_size"], train=False), letterbox=m["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            area_fracs = np.array([
                decode_instance_mask(seg).sum() / (seg["size"][0] * seg["size"][1])
                for _fn, seg, _box, _lbl in ds.instances
            ])
            combined_probs = np.zeros((len(ds.instances), len(class_names)), dtype=np.float64)

        model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        probs = []
        with torch.no_grad():
            for images, _labels in loader:
                logits = model(images.to(device))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs)
        per_member_argmax.append(probs.argmax(axis=1))

        weight = weight_320 if m["label"] == "resnet50_320" else w_rest
        combined_probs += weight * probs
        print(f"scored {m['label']} (weight={weight:.2f})")

    y_pred = combined_probs.argmax(axis=1)
    confidence = combined_probs.max(axis=1)
    correct = y_pred == y_true
    n_total = len(y_true)
    n_errors = int((~correct).sum())
    print(f"\nofficial test: n={n_total}, accuracy={correct.mean():.4f}, errors={n_errors}")

    per_member_argmax = np.stack(per_member_argmax, axis=1)
    agreement = (per_member_argmax == y_pred[:, None]).sum(axis=1)  # how many of 4 members agree with the ensemble

    thresholds = np.arange(0.30, 1.00, 0.05)
    rows = []
    for t in thresholds:
        flagged = confidence < t
        n_flagged = int(flagged.sum())
        errors_caught = int((flagged & ~correct).sum())
        correct_flagged = int((flagged & correct).sum())
        recall = errors_caught / n_errors if n_errors else 0.0
        precision = errors_caught / n_flagged if n_flagged else 0.0

        expected_fixed = errors_caught * FIX_RATE
        expected_broken = correct_flagged * REGRESSION_RATE
        net_accuracy_delta = (expected_fixed - expected_broken) / n_total
        total_cost_s = n_flagged * SAM2_COST_S

        rows.append({
            "confidence_threshold": float(t), "n_flagged": n_flagged, "pct_flagged": n_flagged / n_total * 100,
            "errors_caught": errors_caught, "recall_of_errors": recall, "precision": precision,
            "correct_instances_flagged": correct_flagged,
            "expected_fixed": expected_fixed, "expected_broken": expected_broken,
            "net_accuracy_delta": net_accuracy_delta, "total_cost_s": total_cost_s,
        })
        print(f"t={t:.2f}: flagged={n_flagged:4d} ({n_flagged/n_total*100:5.1f}%)  "
              f"recall={recall*100:5.1f}%  precision={precision*100:5.1f}%  "
              f"net_acc_delta={net_accuracy_delta*100:+.3f}%  cost={total_cost_s/60:.1f}min")

    result = {
        "n_total": n_total, "n_errors": n_errors, "baseline_accuracy": float(correct.mean()),
        "fix_rate_used": FIX_RATE, "regression_rate_used": REGRESSION_RATE, "sam2_cost_s": SAM2_COST_S,
        "thresholds": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
