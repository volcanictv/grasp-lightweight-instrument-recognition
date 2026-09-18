"""Logit-space MC Dropout uncertainty signals for the Task B ensemble, each
scored by ROC-AUC against incorrect=1 / correct=0 (docs/DECISIONS.md
2026-09-17/18). The earlier comparison (analyze_uncertainty_signals.py) only
kept probability-space summaries; this keeps the raw per-pass logits so any
further signal can be computed without re-running the network.

Per member, from S stochastic passes (Dropout in train mode, everything else
eval) of logits z_s in R^C:

  mean_logit_variance      mean over classes of Var_s(z_s[c])
  max_logit_variance       max over classes of Var_s(z_s[c])
  logit_l2_variance        E_s ||z_s - mean_s z||_2^2 (trace of the logit
                           covariance). Equals C * mean_logit_variance, so it
                           is rank-identical to it by construction.
  margin_std               std over passes of the per-pass top1-top2 logit margin
  margin_range             max - min over passes of that margin
  variation_ratio          1 - (count of the modal class among the S argmaxes) / S
  top1_class_switch_rate   fraction of passes whose argmax differs from the
                           member's own deterministic (dropout-off) argmax

Extras, labelled as such: logit_norm_variance (Var_s ||z_s||_2, the other
plausible reading of "L2 variance") and mc_top_class_disagreement_vs_ensemble
(the earlier 0.873 signal, recomputed here as a pipeline sanity check).

Member signals are combined with the ensemble weights. "Incorrect" is the
deterministic ensemble prediction being wrong, same as before. Raw logits are
cached to --logits-cache; if that file exists no model is run.

Usage:
    python scripts/analyze_mc_logit_signals.py --ensemble-config configs/region_ensemble_mcdropout.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

from analyze_uncertainty_signals import enable_mc_dropout
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

SIGNALS_ORDER = [
    "mean_logit_variance", "max_logit_variance", "logit_l2_variance", "margin_std", "margin_range",
    "variation_ratio", "top1_class_switch_rate",
]
EXTRAS = ["logit_norm_variance", "mc_top_class_disagreement_vs_ensemble"]


def collect_logits(m: dict, data_root: Path, class_names: list, device: torch.device, samples: int):
    ds = GraspRegionDataset(
        data_root, "test", transform=build_transforms(m["image_size"], train=False), letterbox=m["letterbox"],
    )
    y_true = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
    model = build_model(m["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    det, mc = [], [[] for _ in range(samples)]
    with torch.no_grad():
        for images, _labels in loader:
            x = images.to(device)
            model.eval()
            det.append(model(x).float().cpu().numpy())
            enable_mc_dropout(model)
            for s in range(samples):
                mc[s].append(model(x).float().cpu().numpy())
    return y_true, np.concatenate(det), np.stack([np.concatenate(p) for p in mc])


def member_signals(det_logits: np.ndarray, mc_logits: np.ndarray, y_pred: np.ndarray) -> dict:
    n_samples, _, n_classes = mc_logits.shape
    var = mc_logits.var(axis=0)  # (N, C)
    centered = mc_logits - mc_logits.mean(axis=0, keepdims=True)

    top2 = np.sort(mc_logits, axis=2)[..., -2:]
    margin = top2[..., 1] - top2[..., 0]  # (S, N)

    pred_s = mc_logits.argmax(axis=2)  # (S, N)
    modal_count = (pred_s[..., None] == np.arange(n_classes)).sum(axis=0).max(axis=1)

    return {
        "mean_logit_variance": var.mean(axis=1),
        "max_logit_variance": var.max(axis=1),
        "logit_l2_variance": (centered ** 2).sum(axis=2).mean(axis=0),
        "margin_std": margin.std(axis=0),
        "margin_range": margin.max(axis=0) - margin.min(axis=0),
        "variation_ratio": 1.0 - modal_count / n_samples,
        "top1_class_switch_rate": (pred_s != det_logits.argmax(axis=1)[None, :]).mean(axis=0),
        "logit_norm_variance": np.linalg.norm(mc_logits, axis=2).var(axis=0),
        "mc_top_class_disagreement_vs_ensemble": (pred_s != y_pred[None, :]).mean(axis=0),
    }


def bootstrap_auroc_ci(is_wrong: np.ndarray, score: np.ndarray, rng: np.random.Generator, n_boot: int = 1000):
    n = len(is_wrong)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if is_wrong[idx].min() == is_wrong[idx].max():
            continue
        aucs.append(roc_auc_score(is_wrong[idx], score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_mcdropout.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "reports" / "mc_logit_signals.json")
    parser.add_argument("--logits-cache", type=Path, default=REPO_ROOT / "experiments" / "mc_logits_cache.npz")
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]
    w_320 = cfg["weight_resnet50_320"]
    weights = np.array([w_320 if m["label"] == "resnet50_320" else (1 - w_320) / (len(members) - 1) for m in members])

    if args.logits_cache.exists():
        cache = np.load(args.logits_cache)
        y_true = cache["y_true"]
        det_logits = [cache[f"det_{m['label']}"] for m in members]
        mc_logits = [cache[f"mc_{m['label']}"] for m in members]
        print(f"loaded cached logits from {args.logits_cache}")
    else:
        class_names = GraspRegionDataset(data_root, "test", transform=build_transforms(224, train=False)).class_names_ordered()
        det_logits, mc_logits, y_true = [], [], None
        for m in members:
            print(f"collecting {args.mc_samples} MC passes: {m['label']}")
            yt, det, mc = collect_logits(m, data_root, class_names, device, args.mc_samples)
            y_true = yt if y_true is None else y_true
            assert (yt == y_true).all(), "member datasets disagree on instance order"
            det_logits.append(det)
            mc_logits.append(mc)
        args.logits_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.logits_cache, y_true=y_true,
            **{f"det_{m['label']}": d for m, d in zip(members, det_logits)},
            **{f"mc_{m['label']}": x for m, x in zip(members, mc_logits)},
        )
        print(f"wrote {args.logits_cache}")

    det_probs = [torch.softmax(torch.from_numpy(d), dim=1).numpy() for d in det_logits]
    combined = (weights[:, None, None] * np.stack(det_probs)).sum(axis=0)
    y_pred = combined.argmax(axis=1)
    is_wrong = (y_pred != y_true).astype(int)
    print(f"baseline: accuracy={1 - is_wrong.mean():.4f}, errors={is_wrong.sum()}/{len(y_true)}")

    per_member = [member_signals(d, x, y_pred) for d, x in zip(det_logits, mc_logits)]
    names = SIGNALS_ORDER + EXTRAS
    combined_signals = {n: sum(w * ms[n] for w, ms in zip(weights, per_member)) for n in names}

    rng = np.random.default_rng(args.seed)
    results = {
        "n_total": int(len(y_true)), "n_errors": int(is_wrong.sum()), "baseline_accuracy": float(1 - is_wrong.mean()),
        "mc_samples": args.mc_samples, "seed": args.seed, "ensemble_config": str(args.ensemble_config.name),
        "auroc": {}, "auroc_ci95": {}, "auroc_per_member": {},
    }
    ref = float(roc_auc_score(is_wrong, -combined.max(axis=1)))
    results["auroc"]["max_softmax_confidence (reference)"] = ref

    print("\nROC-AUC, incorrect=1 vs correct=0 (higher score = more uncertain; 0.5 = chance):")
    for n in names:
        auroc = float(roc_auc_score(is_wrong, combined_signals[n]))
        lo, hi = bootstrap_auroc_ci(is_wrong, combined_signals[n], rng)
        results["auroc"][n] = auroc
        results["auroc_ci95"][n] = [lo, hi]
        results["auroc_per_member"][n] = {
            m["label"]: float(roc_auc_score(is_wrong, ms[n])) for m, ms in zip(members, per_member)
        }
        tag = "" if n in SIGNALS_ORDER else "  (extra)"
        print(f"  {n:<40} {auroc:.4f}  [{lo:.4f}, {hi:.4f}]{tag}")
    print(f"  {'max_softmax_confidence (reference)':<40} {ref:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
