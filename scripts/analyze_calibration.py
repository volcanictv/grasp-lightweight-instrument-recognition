"""Checks whether the ensemble's confidence is actually calibrated
(P(correct) matches the stated probability), not just good at *ranking*
errors (docs/DECISIONS.md 2026-09-16's AUROC comparison answered ranking;
"softmax is overconfident" is usually a calibration complaint instead --
a signal can rank well while being badly calibrated, so this is a
different, complementary check).

Temperature scaling (Guo et al. 2017) is the standard, cheap, reviewer-
accepted fix if miscalibration shows up: rescale each member's logits by
a learned scalar T before softmax (`softmax(logits / T)`), fit by
minimizing NLL on held-out data. Fit per-member (not one shared T),
since the 4 members are different architectures with no reason to share
a temperature.

Split, not the whole official test set for both fitting and evaluating:
this ensemble was trained on `data.split: official`, so fold1/fold2 are
NOT a valid held-out set for it (those cases were in its own training
data) -- unlike every other fold-based check in this project. Splits
official test itself by case instead (never by frame, per CLAUDE.md):
CASE047 + CASE051 (639 instances) to fit temperature, CASE041 + CASE050
+ CASE053 (2,222 instances) to evaluate calibration on -- genuinely
unseen by the fitting step either way.

Usage:
    python scripts/analyze_calibration.py
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

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

FIT_CASES = {"CASE047", "CASE051"}
EVAL_CASES = {"CASE041", "CASE050", "CASE053"}
N_BINS = 15


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, temperature: float, y_true: np.ndarray) -> float:
    probs = softmax_np(logits / temperature)
    return float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    candidates = np.linspace(0.3, 5.0, 200)
    losses = [nll(logits, t, y_true) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict]]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        n_b = int(in_bin.sum())
        if n_b == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "accuracy": None, "confidence": None})
            continue
        acc_b = float(correct[in_bin].mean())
        conf_b = float(confidences[in_bin].mean())
        ece += (n_b / len(confidences)) * abs(acc_b - conf_b)
        bins.append({"lo": float(lo), "hi": float(hi), "n": n_b, "accuracy": acc_b, "confidence": conf_b})
    return float(ece), bins


def main() -> None:
    ensemble_config_path = REPO_ROOT / "configs" / "region_ensemble.yaml"
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_path = REPO_ROOT / "docs" / "reports" / "calibration_analysis.json"

    ensemble_config = yaml.safe_load(ensemble_config_path.read_text())
    members_cfg = ensemble_config["members"]
    weight_320 = ensemble_config["weight_resnet50_320"]
    w_rest = (1 - weight_320) / (len(members_cfg) - 1)
    weights = np.array([weight_320 if m["label"] == "resnet50_320" else w_rest for m in members_cfg])

    class_names = None
    y_true = None
    case_of = None
    per_member_logits = []

    for m in members_cfg:
        ds = GraspRegionDataset(
            data_root, "test", transform=build_transforms(m["image_size"], train=False), letterbox=m["letterbox"],
        )
        if class_names is None:
            class_names = ds.class_names_ordered()
            y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            case_of = np.array([fn.split("/")[0] for fn, _seg, _box, _lbl in ds.instances])

        model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()

        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
        logits = []
        with torch.no_grad():
            for images, _labels in loader:
                logits.append(model(images.to(device)).cpu().numpy())
        per_member_logits.append(np.concatenate(logits))
        print(f"scored {m['label']}")

    fit_mask = np.isin(case_of, list(FIT_CASES))
    eval_mask = np.isin(case_of, list(EVAL_CASES))
    print(f"\nfit set: {fit_mask.sum()} instances ({sorted(FIT_CASES)}), eval set: {eval_mask.sum()} instances ({sorted(EVAL_CASES)})")

    temperatures = []
    for i, m in enumerate(members_cfg):
        t = fit_temperature(per_member_logits[i][fit_mask], y_true[fit_mask])
        temperatures.append(t)
        print(f"  {m['label']}: fitted temperature T={t:.3f}")

    raw_probs = sum(w * softmax_np(logits) for w, logits in zip(weights, per_member_logits))
    calibrated_probs = sum(w * softmax_np(logits / t) for w, logits, t in zip(weights, per_member_logits, temperatures))

    def summarize(probs: np.ndarray, mask: np.ndarray) -> dict:
        y_pred = probs.argmax(axis=1)
        correct = (y_pred == y_true).astype(int)
        confidence = probs.max(axis=1)
        acc = float(correct[mask].mean())
        ece, bins = expected_calibration_error(confidence[mask], correct[mask])
        return {"accuracy": acc, "ece": ece, "bins": bins}

    raw_summary = summarize(raw_probs, eval_mask)
    calibrated_summary = summarize(calibrated_probs, eval_mask)

    print(f"\neval set (n={eval_mask.sum()}), uncalibrated (T=1): accuracy={raw_summary['accuracy']:.4f}  ECE={raw_summary['ece']:.4f}")
    print(f"eval set (n={eval_mask.sum()}), temperature-scaled:  accuracy={calibrated_summary['accuracy']:.4f}  ECE={calibrated_summary['ece']:.4f}")

    results = {
        "fit_cases": sorted(FIT_CASES), "eval_cases": sorted(EVAL_CASES),
        "n_fit": int(fit_mask.sum()), "n_eval": int(eval_mask.sum()),
        "fitted_temperatures": {m["label"]: t for m, t in zip(members_cfg, temperatures)},
        "uncalibrated": raw_summary, "temperature_scaled": calibrated_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
