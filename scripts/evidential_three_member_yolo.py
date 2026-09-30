"""Three-member evidential pipeline (resnet50_224 dropped, weights 0.40 / 0.30 / 0.30 over
resnet50_320, baseline, letterbox_crop) with the YOLO26-seg tracker. The members are the
already-trained evidential ones; only the gate threshold and the tracker change.

Stages, each writing next to --work-dir:
    prep-fold1     rank fold1 instances by the 3-member epistemic score S1 and write the index
                   files the YOLO tracker must process (every instance above the lowest grid
                   threshold), one file per shard
    calibrate      fold1 threshold by the same rule as evidential_gate_fold1_calibrate.py:
                   maximise accuracy, then the highest threshold within 0.0010 of the best
    prep-official  apply that threshold unchanged to the official test, write index files for the
                   flagged set plus the top-833 set so budget-matched rows can be scored too
    score          official-test accuracy/macro-F1 at the threshold and at 539 / 833 tracked

Tracker outputs (scripts/evaluate_temporal_track_yolo.py --frame-logits-out) go in
<work-dir>/fold1_frames_shard<n>.npz and <work-dir>/official_frames_shard<n>.npz.
scripts/run_three_member_yolo.sh chains all of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import yaml
from sklearn.metrics import f1_score

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

WEIGHTS = {"resnet50_320": 0.40, "baseline": 0.30, "letterbox_crop": 0.30}
TOLERANCE = 0.0010
PERCENTILES = [98, 96, 94, 92, 90, 88, 85, 80, 75, 70, 65]
BUDGETS = (539, 833)
LABELS = list(range(7))


def member_order(ensemble_config: Path) -> list[str]:
    return [m["label"] for m in yaml.safe_load(ensemble_config.read_text())["members"]]


def base_scores(extract: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(extract)
    alpha = sum(w * alpha_from_logits(d[f"det_{k}"]) for k, w in WEIGHTS.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    return d["y"], mu.argmax(axis=1), variance_scores(alpha)["epistemic"]


def track_predict(det_frames: np.ndarray, order: list[str]) -> int:
    """det_frames: (frames, members in `order`, classes) raw logits. Frame weight is the frame's
    top mu_bar, final class the argmax of the weighted sum (docs/DECISIONS.md 2026-09-28)."""
    alpha = sum(w * alpha_from_logits(det_frames[:, order.index(k), :]) for k, w in WEIGHTS.items())
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    return int((mu.max(axis=1)[:, None] * mu).sum(axis=0).argmax())


def load_tracked(work_dir: Path, prefix: str, order: list[str]) -> dict[int, int]:
    tracked = {}
    for path in sorted(work_dir.glob(f"{prefix}_frames_shard*.npz")):
        z = np.load(path)
        for key in z.files:
            if key.startswith("det_"):
                tracked[int(key[4:])] = track_predict(z[key], order)
    return tracked


def write_shards(work_dir: Path, name: str, indices: np.ndarray, num_shards: int) -> None:
    for shard in range(num_shards):
        (work_dir / f"{name}_track_shard{shard}.json").write_text(
            json.dumps({"errors": [{"index": int(i)} for i in indices[shard::num_shards]]}))


def apply_gate(base: np.ndarray, tracked: dict[int, int], flag: np.ndarray) -> np.ndarray:
    pred = base.copy()
    for i in np.where(flag)[0]:
        if int(i) in tracked:
            pred[i] = tracked[int(i)]
    return pred


def row(y: np.ndarray, base: np.ndarray, pred: np.ndarray, flag: np.ndarray, **extra) -> dict:
    return {**extra, "tracked": int(flag.sum()), "share": float(flag.mean()),
            "accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS)),
            "fixed": int(((base != y) & (pred == y) & flag).sum()), "broken": int(((base == y) & (pred != y) & flag).sum())}


def prep_fold1(args: argparse.Namespace) -> None:
    y, base, s1 = base_scores(args.extract_dir / "E_grasp_fold1_s42.npz")
    grid = sorted({round(float(x), 6) for x in np.percentile(s1, PERCENTILES)}, reverse=True)
    idx = np.where(s1 >= min(grid))[0]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write_shards(args.work_dir, "fold1", idx, args.num_shards)
    print(f"fold1 3-member base accuracy {(base == y).mean():.4f}; {len(idx)} of {len(y)} instances to track")


def calibrate(args: argparse.Namespace) -> None:
    order = member_order(args.fold1_config)
    y, base, s1 = base_scores(args.extract_dir / "E_grasp_fold1_s42.npz")
    tracked = load_tracked(args.work_dir, "fold1", order)
    grid = sorted({round(float(x), 6) for x in np.percentile(s1, PERCENTILES)}, reverse=True)
    missing = set(np.where(s1 >= min(grid))[0].tolist()) - set(tracked)
    if missing:
        raise SystemExit(f"{len(missing)} flagged fold1 instances have no tracking yet")
    rows = {}
    for t in grid:
        flag = s1 >= t
        rows[str(t)] = row(y, base, apply_gate(base, tracked, flag), flag, threshold=t)
        r = rows[str(t)]
        print(f"t={t:.6f} tracked={r['tracked']:4d} ({100 * r['share']:.1f}%) acc={r['accuracy']:.4f} "
              f"macroF1={r['macro_f1']:.4f} fixed={r['fixed']} broken={r['broken']}")
    best = max(r["accuracy"] for r in rows.values())
    chosen = max(r["threshold"] for r in rows.values() if r["accuracy"] >= best - TOLERANCE)
    out = {"members": list(WEIGHTS.items()), "base_accuracy": float((base == y).mean()), "n": int(len(y)),
           "rows": rows, "chosen_threshold": chosen, "chosen_row": rows[str(chosen)]}
    (args.work_dir / "fold1_calibration.json").write_text(json.dumps(out, indent=1))
    print(f"chosen threshold {chosen:.6f}: {rows[str(chosen)]}")


def prep_official(args: argparse.Namespace) -> None:
    tau = json.loads((args.work_dir / "fold1_calibration.json").read_text())["chosen_threshold"]
    y, base, s1 = base_scores(args.extract_dir / "E_grasp_official_s42.npz")
    flagged = np.where(s1 >= tau)[0]
    top = np.argsort(-s1, kind="stable")[:max(BUDGETS)]
    union = np.array(sorted(set(flagged.tolist()) | set(top.tolist())))
    write_shards(args.work_dir, "official", union, args.num_shards)
    print(f"official 3-member base accuracy {(base == y).mean():.4f}; tau={tau:.6f} flags {len(flagged)} "
          f"({100 * len(flagged) / len(y):.1f}%); tracking {len(union)} instances (threshold set plus top-{max(BUDGETS)})")


def score(args: argparse.Namespace) -> None:
    order = member_order(args.official_config)
    tau = json.loads((args.work_dir / "fold1_calibration.json").read_text())["chosen_threshold"]
    y, base, s1 = base_scores(args.extract_dir / "E_grasp_official_s42.npz")
    tracked = load_tracked(args.work_dir, "official", order)
    rank = np.argsort(-s1, kind="stable")
    plans = {"threshold": s1 >= tau}
    for k in BUDGETS:
        flag = np.zeros(len(y), bool)
        flag[rank[:k]] = True
        plans[f"top{k}"] = flag
    results = {"tau": tau, "base_accuracy": float((base == y).mean()),
               "base_macro_f1": float(f1_score(y, base, average="macro", labels=LABELS)), "rows": {}}
    for name, flag in plans.items():
        missing = int(sum(1 for i in np.where(flag)[0] if int(i) not in tracked))
        if missing:
            results["rows"][name] = {"scored": False, "missing_tracking": missing}
            continue
        results["rows"][name] = {"scored": True, **row(y, base, apply_gate(base, tracked, flag), flag)}
    (args.work_dir / "official_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prep-fold1", "calibrate", "prep-official", "score"])
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "experiments_edl" / "extract")
    ap.add_argument("--fold1-config", type=Path, default=REPO_ROOT / "configs" / "evidential" / "ens_E_fold1_s42_lam0p01a10.yaml")
    ap.add_argument("--official-config", type=Path, default=REPO_ROOT / "configs" / "evidential" / "ens_E_official_s42_lam0p01a10.yaml")
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    {"prep-fold1": prep_fold1, "calibrate": calibrate, "prep-official": prep_official, "score": score}[args.stage](args)


if __name__ == "__main__":
    main()
