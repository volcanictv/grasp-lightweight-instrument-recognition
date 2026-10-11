"""Writes a boxes.json for the handoff package's command-line runner from the ground-truth boxes of a few official-test frames of one case,
plus truth.json with those instruments' labels, so the runner's output can be compared with the truth.

Usage (titanxp, surgical environment):
    python scripts/make_cli_smoke_inputs.py --case CASE050 --frames 4 --out ~/cli_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from surgical_ai.data.region_dataset import GraspRegionDataset

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="CASE050")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    args = ap.parse_args()
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    by_frame = defaultdict(list)
    for file_name, _seg, box, label in ds.instances:
        if file_name.startswith(args.case + "/"):
            by_frame[file_name.split("/")[1]].append((list(map(float, box)), CLASSES[label]))
    names = sorted(by_frame)
    pick = [names[i] for i in np.linspace(0, len(names) - 1, args.frames).round().astype(int)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "boxes.json").write_text(json.dumps({n: [b for b, _ in by_frame[n]] for n in pick}, indent=1))
    (args.out / "truth.json").write_text(json.dumps({n: [c for _, c in by_frame[n]] for n in pick}, indent=1))
    print(f"wrote {args.out}/boxes.json and truth.json: {len(pick)} frames of {args.case}, {sum(len(by_frame[n]) for n in pick)} instruments")


if __name__ == "__main__":
    main()
