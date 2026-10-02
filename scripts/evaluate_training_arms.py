"""Scores the training-experiment arms against the baseline and applies the decision rule of
docs/reports/training_experiments_preregistration.md: an arm improves only if its mean accuracy beats the baseline mean by
more than the larger of the two seed standard deviations.

Per arm and seed: three-member ensemble alone on the held-out fold (accuracy, macro-F1), and accuracy on the held-out
fold's tracker-style neighbour crops (scripts/eval_on_neighbour_crops.py).

Usage (titanxp, repo root, surgical environment):
    python scripts/evaluate_training_arms.py --fold fold1 --arms baseline P N --seeds 42 43 44 \\
        --heldout-neighbour-dir experiments/temporal_neighbors/fold1 --data-root <GraSP> --out docs/reports/training_arms/fold1.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import evidential_make_configs as mk
import evidential_three_member_yolo as t
from run_training_arms import MEMBERS, arm_stem

EXP = REPO_ROOT / "experiments"


def ensemble_config(arm: str, fold: str, seed: int) -> Path:
    if arm == "baseline":
        return REPO_ROOT / "configs" / "evidential" / f"ens_E_{fold}_s{seed}_lam0p01a10.yaml"
    members = []
    for label in MEMBERS:
        runs = sorted(EXP.glob(f"{arm_stem(arm, label, fold, seed)}_2*"))
        runs = [r for r in runs if (r / "best.pt").exists()]
        if not runs:
            raise FileNotFoundError(f"no finished run for {arm_stem(arm, label, fold, seed)}")
        m = mk.MEMBERS[label]
        members.append({"checkpoint": str((runs[-1] / "best.pt").relative_to(REPO_ROOT)), "model": m["e"], "image_size": m["size"],
                        "letterbox": m["letterbox"], "label": label})
    path = REPO_ROOT / "configs" / "arms" / f"ens_{arm}_{fold}_s{seed}.yaml"
    path.write_text(yaml.safe_dump({"weight_resnet50_320": mk.WEIGHT_320, "members": members}, sort_keys=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", choices=["fold1", "fold2"], required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--heldout-neighbour-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (EXP / "arms_extract").mkdir(exist_ok=True)
    rows: dict[str, dict[int, dict]] = {}
    for arm in args.arms:
        for seed in args.seeds:
            try:
                cfg = ensemble_config(arm, args.fold, seed)
            except FileNotFoundError as e:
                print("skip", arm, seed, e)
                continue
            if arm == "baseline":
                extract = REPO_ROOT / "experiments_edl" / "extract" / f"E_grasp_{args.fold}_s{seed}.npz"
            else:
                extract = EXP / "arms_extract" / f"{arm}_{args.fold}_s{seed}.npz"
                if not extract.exists():
                    subprocess.run([sys.executable, "scripts/evidential_extract.py", "--ensemble-config", str(cfg), "--target", f"grasp_{args.fold}",
                                    "--data-root", str(args.data_root), "--device", args.device, "--out", str(extract)], check=True, cwd=REPO_ROOT)
            y, pred, _s1 = t.base_scores(extract)
            crops_out = EXP / "arms_extract" / f"{arm}_{args.fold}_s{seed}_crops.json"
            if not crops_out.exists():
                subprocess.run([sys.executable, "scripts/eval_on_neighbour_crops.py", "--ensemble-config", str(cfg), "--neighbour-dir",
                                str(args.heldout_neighbour_dir), "--json-split", args.fold, "--data-root", str(args.data_root),
                                "--device", args.device, "--out", str(crops_out)], check=True, cwd=REPO_ROOT)
            crops = json.loads(crops_out.read_text())
            from sklearn.metrics import f1_score
            rows.setdefault(arm, {})[seed] = {"accuracy": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", labels=range(7))),
                                              "crop_accuracy": crops["accuracy"], "crop_macro_f1": crops["macro_f1"]}

    def stat(arm: str, key: str) -> tuple[float, float, int]:
        v = np.array([r[key] for r in rows.get(arm, {}).values()])
        return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)) if len(v) else (float("nan"), 0.0, 0)

    summary = {}
    print(f"{'arm':<10}{'seeds':>6}{'acc':>16}{'macro-F1':>16}{'crop acc':>16}   decision vs baseline (accuracy)")
    for arm in args.arms:
        a, f, c = stat(arm, "accuracy"), stat(arm, "macro_f1"), stat(arm, "crop_accuracy")
        base = stat("baseline", "accuracy")
        delta = a[0] - base[0]
        verdict = "baseline" if arm == "baseline" else ("IMPROVEMENT" if delta > max(a[1], base[1]) else "no resolvable effect" if delta > -max(a[1], base[1]) else "WORSE")
        summary[arm] = {"seeds": a[2], "accuracy": a, "macro_f1": f, "crop_accuracy": c, "delta_accuracy_vs_baseline": delta, "verdict": verdict}
        print(f"{arm:<10}{a[2]:>6}{a[0]:>10.4f}+-{a[1]:.4f}{f[0]:>10.4f}+-{f[1]:.4f}{c[0]:>10.4f}+-{c[1]:.4f}   {verdict} ({delta:+.4f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"fold": args.fold, "per_seed": {a: {str(s): r for s, r in d.items()} for a, d in rows.items()}, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
