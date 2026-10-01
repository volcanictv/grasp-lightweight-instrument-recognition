"""Collects the fold1 sweep of YOLO tracker matching options (scripts/sweep_yolo_tracker_fold1.sh)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    print(f"{'config':<24}{'mean_len':>9}{'center_only':>12}{'tau':>10}{'tracked%':>9}{'acc':>8}{'macroF1':>9}{'fixed':>6}{'broken':>7}")
    base = None
    for d in sorted(p for p in args.dir.iterdir() if p.is_dir()):
        try:
            inst = json.loads((d / "fold1_tracked_shard0.json").read_text())["instances"]
            cal = json.loads((d / "fold1_calibration.json").read_text())
        except FileNotFoundError:
            print(f"{d.name:<24}incomplete")
            continue
        base = cal["base_accuracy"]
        lens = [r["track_len"] for r in inst]
        c = cal["chosen_row"]
        print(f"{d.name:<24}{sum(lens) / len(lens):>9.1f}{sum(n == 1 for n in lens) / len(lens):>12.3f}{cal['chosen_threshold']:>10.6f}"
              f"{100 * c['share']:>9.1f}{c['accuracy']:>8.4f}{c['macro_f1']:>9.4f}{c['fixed']:>6}{c['broken']:>7}")
    print("fold1 base accuracy", base)


if __name__ == "__main__":
    main()
