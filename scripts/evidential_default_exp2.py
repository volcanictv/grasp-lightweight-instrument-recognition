"""Experiment 2 driver (docs/DECISIONS.md 2026-09-29): train the evidential official-split
ensemble with seeds 43 and 44 (identical config to seed 42: lambda 0.01, KL anneal 10 of 20),
then extract single-pass official-test logits. Resumable: skips finished trainings/extractions.
Run on titanxp from the repo root, two GPUs, two dataloader workers per job.

Usage:
    python scripts/evidential_default_exp2.py train
    python scripts/evidential_default_exp2.py extract
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evidential_make_configs as mk

PY = "/home/yzx/miniconda3/envs/surgical/bin/python"
EXP = REPO_ROOT / "experiments"
OUT = REPO_ROOT / "experiments_edl" / "extract"
LAM, ANNEAL = 0.01, 10
SEEDS = (43, 44)
ORDER = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]


def finished(stem: str) -> bool:
    return any((d / "best.pt").exists() and (d / "manifest.json").exists() for d in sorted(EXP.glob(f"{stem}_2*")))


def train() -> None:
    tag = mk.lam_tag(LAM, ANNEAL)
    jobs = []
    for member in ORDER:
        for seed in SEEDS:
            s = mk.stem("E", member, "official", seed, tag)
            (mk.OUT / f"{s}.yaml").write_text(yaml.safe_dump(
                mk.train_config("E", member, "official", seed, LAM, ANNEAL), sort_keys=False))
            if not finished(s):
                jobs.append(s)
    print(len(jobs), "trainings to run", flush=True)
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    (EXP / "evidential_default").mkdir(parents=True, exist_ok=True)

    def worker(gpu: int) -> None:
        while True:
            try:
                s = q.get_nowait()
            except queue.Empty:
                return
            with open(EXP / "evidential_default" / f"train_{s}.log", "w") as log:
                subprocess.run([PY, "scripts/train.py", f"configs/evidential/{s}.yaml", "--data-root", "GraSP",
                                "--device", f"cuda:{gpu}", "--num-workers", "2"], stdout=log,
                               stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            print("done", s, flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def extract() -> None:
    tag = mk.lam_tag(LAM, ANNEAL)
    OUT.mkdir(parents=True, exist_ok=True)
    plans = []
    for seed in SEEDS:
        subprocess.run([PY, "scripts/evidential_make_configs.py", "ensemble", "--arm", "E", "--fold", "official",
                        "--seed", str(seed), "--lam", str(LAM), "--anneal", str(ANNEAL)], check=True, cwd=REPO_ROOT)
        plans.append((seed, f"configs/evidential/ens_E_official_s{seed}_{tag}.yaml"))
    procs = []
    for gpu, (seed, cfg) in enumerate(plans):
        out = OUT / f"E_grasp_official_s{seed}.npz"
        if out.exists():
            continue
        log = open(OUT / f"E_grasp_official_s{seed}.log", "w")
        procs.append(subprocess.Popen([PY, "scripts/evidential_extract.py", "--ensemble-config", cfg, "--target",
                                       "grasp_test", "--data-root", "GraSP", "--mc-samples", "0", "--device",
                                       f"cuda:{gpu}", "--out", str(out)], stdout=log, stderr=subprocess.STDOUT,
                                      cwd=REPO_ROOT))
    for p in procs:
        p.wait()


if __name__ == "__main__":
    {"train": train, "extract": extract}[sys.argv[1]]()
