"""Mean and seed spread of the per-seed results of scripts/gtbox_sam_final_eval.py (docs/reports/gtbox_sam/<variant>_s<seed>.json).

Usage:
    python scripts/aggregate_gtbox_seeds.py --variant finetuned --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    folder = REPO_ROOT / "docs" / "reports" / "gtbox_sam"
    runs = [json.loads((folder / f"{args.variant}_s{s}.json").read_text()) for s in args.seeds]
    names = [k for k, v in runs[0].items() if isinstance(v, dict) and "mIoU" in v]
    summary = {"seeds": args.seeds, "configs": {}}
    print(f"{'configuration':<24}{'mIoU':>14}{'IoU':>14}{'mcIoU':>14}{'inst acc':>16}{'macro-F1':>16}")
    for name in names:
        row = {}
        for key in ("mIoU", "IoU", "mcIoU", "instance_accuracy", "instance_macro_f1"):
            v = np.array([r[name][key] for r in runs])
            row[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}
        row["per_class_iou"] = {c: float(np.mean([r[name]["per_class_iou"][c] for r in runs])) for c in runs[0][name]["per_class_iou"]}
        row["tracked"] = runs[0][name].get("tracked")
        summary["configs"][name] = row
        f = lambda k, scale=100: f"{scale * row[k]['mean']:.2f}+-{scale * row[k]['sd']:.2f}"
        print(f"{name:<24}{f('mIoU'):>14}{f('IoU'):>14}{f('mcIoU'):>14}{f('instance_accuracy', 1):>16}{f('instance_macro_f1', 1):>16}")
    summary["published_test"] = runs[0]["published_test"]
    summary["published_crossval"] = runs[0]["published_crossval"]
    (folder / f"{args.variant}_3seed.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
