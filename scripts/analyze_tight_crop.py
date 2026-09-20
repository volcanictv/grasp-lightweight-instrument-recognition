"""Summarises the tight-crop experiment runs (docs/reports/tight_crop/).

Reads each run's manifest.json (config, best-epoch metrics) and its training
log (per-epoch validation macro-F1, which the manifest does not keep), groups
runs into conditions {crop_variant} x {augmentation} x split, and reports per
condition (mean +/- sd over seeds when there are several):

  best_f1        max validation macro-F1 over epochs (this is what the manifest
                 stores; it is selected on the validation split, so it is
                 optimistic, equally for every condition)
  final_f1       mean of the last 3 epochs, no selection
  curve_mean_f1  mean validation macro-F1 over all epochs (area under the
                 learning curve, a learning-speed summary)
  epochs_to_90   first epoch whose validation macro-F1 reaches 90% of the
                 baseline condition's final_f1 (baseline = standard crop +
                 default augmentation, same split and seed)
  best_acc / per-class F1 (Grasper, Suction) at the best epoch

Usage:
    python scripts/analyze_tight_crop.py --experiments DIR --logs DIR --out-dir OUT
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPOCH_RE = re.compile(r"epoch (\d+)/(\d+) train_loss=([\d.]+) val_loss=([\d.]+) train_macro_f1=([\d.]+) val_macro_f1=([\d.]+)")
BASELINE = ("standard", "default")


def parse_curve(log_path: Path) -> list[float]:
    curve = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            curve[int(m.group(1))] = float(m.group(6))
    return [curve[k] for k in sorted(curve)]


def find_log(logs_dir: Path, config_stem: str, seed: int) -> Path | None:
    for path in sorted(logs_dir.glob(f"{config_stem}*.log")):
        if path.stem.endswith(f"_s{seed}"):
            return path
    return None


def collect(experiments: Path, logs: Path) -> list[dict]:
    runs = []
    for manifest_path in sorted(experiments.glob("*/manifest.json")):
        m = json.loads(manifest_path.read_text())
        cfg, data = m["config"], m["config"]["data"]
        stem = Path(m["config_path"]).stem
        log = find_log(logs, stem, cfg["training"]["seed"])
        curve = parse_curve(log) if log else []
        runs.append({
            "run_id": m["run_id"], "split": data["split"], "seed": cfg["training"]["seed"],
            "crop": data.get("crop_variant", "standard"), "aug": data.get("augmentation", "default"),
            "curve": curve, "acc": m["final_metrics"]["accuracy"], "best_f1": m["final_metrics"]["macro_f1"],
            "per_class_f1": m["final_metrics"]["per_class_f1"], "seconds": m["wall_clock_seconds"],
        })
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = collect(args.experiments, args.logs)
    if not runs:
        raise SystemExit("no finished runs found")

    base_final = {}
    for r in runs:
        if (r["crop"], r["aug"]) == BASELINE and len(r["curve"]) >= 3:
            base_final[(r["split"], r["seed"])] = float(np.mean(r["curve"][-3:]))
    for r in runs:
        c = np.array(r["curve"])
        r["final_f1"] = float(c[-3:].mean()) if len(c) >= 3 else float("nan")
        r["curve_mean_f1"] = float(c.mean()) if len(c) else float("nan")
        target = 0.9 * base_final.get((r["split"], r["seed"]), float("nan"))
        hit = np.where(c >= target)[0] if len(c) else []
        r["epochs_to_90"] = int(hit[0]) + 1 if len(hit) else None

    groups = defaultdict(list)
    for r in runs:
        groups[(r["split"], r["crop"], r["aug"])].append(r)

    def ms(values):
        v = [x for x in values if x is not None and not np.isnan(x)]
        if not v:
            return "n/a"
        return f"{np.mean(v):.4f}" + (f" +/- {np.std(v, ddof=1):.4f}" if len(v) > 1 else "")

    lines = ["| split | crop | aug | seeds | best F1 | final F1 (last 3) | curve mean F1 | epochs to 90% | best acc | Grasper F1 | Suction F1 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for (split, crop, aug), rs in sorted(groups.items()):
        e90 = [r["epochs_to_90"] for r in rs if r["epochs_to_90"] is not None]
        row = {
            "seeds": sorted(r["seed"] for r in rs), "best_f1": [r["best_f1"] for r in rs], "final_f1": [r["final_f1"] for r in rs],
            "curve_mean_f1": [r["curve_mean_f1"] for r in rs], "epochs_to_90": [r["epochs_to_90"] for r in rs],
            "best_acc": [r["acc"] for r in rs],
            "grasper_f1": [r["per_class_f1"].get("Laparoscopic Grasper") for r in rs],
            "suction_f1": [r["per_class_f1"].get("Suction Instrument") for r in rs],
        }
        summary[f"{split}|{crop}|{aug}"] = row
        lines.append(
            f"| {split} | {crop} | {aug} | {len(rs)} | {ms(row['best_f1'])} | {ms(row['final_f1'])} | {ms(row['curve_mean_f1'])} | "
            f"{ms(e90) if e90 else 'never'} | {ms(row['best_acc'])} | {ms(row['grasper_f1'])} | {ms(row['suction_f1'])} |"
        )
    table = "\n".join(lines)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))
    (args.out_dir / "summary.md").write_text(table + "\n")
    print(table)

    fig, axes = plt.subplots(1, len({r["split"] for r in runs}), figsize=(6.5 * len({r["split"] for r in runs}), 4), squeeze=False)
    for ax, split in zip(axes[0], sorted({r["split"] for r in runs})):
        for (sp, crop, aug), rs in sorted(groups.items()):
            if sp != split:
                continue
            curves = [r["curve"] for r in rs if r["curve"]]
            n = min(len(c) for c in curves)
            mean = np.mean([c[:n] for c in curves], axis=0)
            ax.plot(range(1, n + 1), mean, label=f"{crop} + {aug} (n={len(curves)})")
        ax.set_title(f"validation macro-F1 by epoch, {split}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("macro-F1")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "learning_curves.png", dpi=130)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
