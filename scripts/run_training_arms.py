"""Trains the three ensemble members for the training-experiment arms (docs/reports/training_experiments_preregistration.md)
from the baseline evidential configs plus the new data keys, on both GPUs from a shared queue. Resumable: finished
runs (best.pt and manifest.json present) are skipped.

    P  data.mask_perturb_prob 0.5
    N  data.neighbour_dir <training-case neighbour crops>, data.neighbour_prob 0.5
    PN both

Usage (titanxp, repo root):
    python scripts/run_training_arms.py --fold fold1 --arms P N --seeds 42 43 44 --neighbour-dir experiments/temporal_neighbors/fold2
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evidential_make_configs as mk

MEMBERS = ["resnet50_320", "baseline", "letterbox_crop"]
PROB = 0.5
OUT = REPO_ROOT / "configs" / "arms"
EXP = REPO_ROOT / "experiments"


def arm_stem(arm: str, member: str, fold: str, seed: int) -> str:
    return f"arm{arm}_{member}_{fold}_s{seed}"


def arm_config(arm: str, member: str, fold: str, seed: int, neighbour_dir: str | list | None, yolo_neighbour_dir: str | list | None = None) -> dict:
    cfg = mk.train_config("E", member, fold, seed, 0.01, 10)
    if fold == "official":
        # fixed schedule: last-epoch weights, validated on a split inside the training data, so the test cases
        # are never touched while training
        cfg["training"]["select_best"] = False
        cfg["data"]["val_split_override"] = "fold1"
    if "P" in arm:
        cfg["data"]["mask_perturb_prob"] = PROB

    def as_list(v):
        return [] if v is None else ([v] if isinstance(v, str) else list(v))

    dirs = []
    if "N" in arm:  # N: EdgeTAM neighbours; NY: EdgeTAM and YOLO neighbours mixed
        dirs += as_list(neighbour_dir)
    if "Y" in arm:  # Y: YOLO-tracker neighbours only
        dirs += as_list(yolo_neighbour_dir)
    if dirs:
        cfg["data"]["neighbour_dir"] = dirs[0] if len(dirs) == 1 else dirs
        cfg["data"]["neighbour_prob"] = PROB
    return cfg


def finished(stem: str) -> bool:
    return any((d / "best.pt").exists() and (d / "manifest.json").exists() for d in EXP.glob(f"{stem}_2*"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", choices=["fold1", "fold2", "official"], required=True)
    ap.add_argument("--arms", nargs="+", required=True, choices=["P", "N", "PN", "Y", "NY"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--neighbour-dir", nargs="+", default=None, help="neighbour crops of the TRAINING cases (several dirs for official)")
    ap.add_argument("--yolo-neighbour-dir", nargs="+", default=None, help="YOLO-tracker neighbour crops of the training cases (arms Y, NY)")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--python", default="/home/yzx/miniconda3/envs/surgical/bin/python")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    for arm in args.arms:
        for seed in args.seeds:
            for member in MEMBERS:
                stem = arm_stem(arm, member, args.fold, seed)
                (OUT / f"{stem}.yaml").write_text(yaml.safe_dump(arm_config(arm, member, args.fold, seed, args.neighbour_dir, args.yolo_neighbour_dir), sort_keys=False))
                if not finished(stem):
                    jobs.append((member, stem))
    jobs.sort(key=lambda j: MEMBERS.index(j[0]))  # the long ResNet-320 jobs first for balance
    print(len(jobs), "trainings to run", flush=True)
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu: int) -> None:
        while True:
            try:
                _member, stem = q.get_nowait()
            except queue.Empty:
                return
            with open(EXP / f"{stem}.log", "w") as log:
                subprocess.run([args.python, "scripts/train.py", f"configs/arms/{stem}.yaml", "--data-root", str(args.data_root),
                                "--device", f"cuda:{gpu}", "--num-workers", "2"], stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            print("done", stem, flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
