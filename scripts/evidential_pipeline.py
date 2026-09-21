"""Driver for the evidential validation on titanxp (docs/DECISIONS.md 2026-09-21):
choose lambda from the finished lambda-grid runs (held-out macro-F1 only), generate every
config, train all remaining members on two GPUs from a shared queue, write the ensemble
configs, and extract logits for every arm/target. Resumable: skips finished trainings.

Usage (on titanxp, from the repo root):
    python scripts/evidential_pipeline.py choose        # prints and stores the chosen lambda
    python scripts/evidential_pipeline.py train         # all remaining trainings, 2 GPUs
    python scripts/evidential_pipeline.py extract       # ensemble configs + logits
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evidential_make_configs as mk

PY = "/home/yzx/miniconda3/envs/surgical/bin/python"
EXP = REPO_ROOT / "experiments"
OUT = REPO_ROOT / "experiments_edl"
GRID = [(0.01, 10), (0.1, 10), (1.0, 10), (0.142857, 0)]
CHOSEN = OUT / "chosen_lambda.json"


def run_dirs(stem: str) -> list[Path]:
    return sorted(EXP.glob(f"{stem}_2*"))


def finished(stem: str) -> bool:
    return any((d / "best.pt").exists() and (d / "manifest.json").exists() for d in run_dirs(stem))


def choose() -> None:
    rows = []
    for lam, anneal in GRID:
        s = mk.stem("E", "resnet50_320", "fold1", 42, mk.lam_tag(lam, anneal))
        d = [d for d in run_dirs(s) if (d / "manifest.json").exists()]
        assert d, f"lambda run not finished: {s}"
        fm = json.loads((d[-1] / "manifest.json").read_text())["final_metrics"]
        rows.append({"lam": lam, "anneal": anneal, "stem": s, "macro_f1": fm["macro_f1"], "accuracy": fm.get("accuracy")})
    best = max(rows, key=lambda r: r["macro_f1"])
    OUT.mkdir(exist_ok=True)
    CHOSEN.write_text(json.dumps({"grid": rows, "chosen": best}, indent=1))
    print(json.dumps({"grid": rows, "chosen": best}, indent=1))


def load_chosen() -> tuple[float, int, str]:
    c = json.loads(CHOSEN.read_text())["chosen"]
    return c["lam"], c["anneal"], mk.lam_tag(c["lam"], c["anneal"])


def job_list(lam: float, anneal: int) -> list[str]:
    jobs = []
    order = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
    tag = mk.lam_tag(lam, anneal)
    specs = [("E", "fold1", 42), ("E", "fold1", 43), ("E", "fold1", 44), ("C", "fold1", 43), ("C", "fold1", 44),
             ("E", "fold2", 42), ("C", "fold2", 42)]
    for arm, fold, seed in specs:
        for member in order:
            jobs.append((arm, member, fold, seed))
    jobs.sort(key=lambda j: order.index(j[1]))  # long ResNet-320 jobs first for balance
    out = []
    for arm, member, fold, seed in jobs:
        s = mk.stem(arm, member, fold, seed, tag)
        cfg = mk.OUT / f"{s}.yaml"
        cfg.write_text(__import__("yaml").safe_dump(mk.train_config(arm, member, fold, seed, lam, anneal), sort_keys=False))
        if not finished(s):
            out.append(s)
    return out


def train() -> None:
    lam, anneal, _ = load_chosen()
    jobs = job_list(lam, anneal)
    print(len(jobs), "trainings to run", flush=True)
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu: int) -> None:
        while True:
            try:
                s = q.get_nowait()
            except queue.Empty:
                return
            with open(EXP / f"{s}.log", "w") as log:
                subprocess.run([PY, "scripts/train.py", f"configs/evidential/{s}.yaml", "--data-root", "GraSP",
                                "--device", f"cuda:{gpu}", "--num-workers", "2"], stdout=log, stderr=subprocess.STDOUT,
                               cwd=REPO_ROOT)
            print("done", s, flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def extract() -> None:
    lam, anneal, tag = load_chosen()
    (OUT / "extract").mkdir(parents=True, exist_ok=True)
    plans = []  # (arm, fold, seed, ensemble config path)
    for arm, fold, seed in [("E", "fold1", 42), ("E", "fold1", 43), ("E", "fold1", 44), ("C", "fold1", 43),
                            ("C", "fold1", 44), ("E", "fold2", 42), ("C", "fold2", 42)]:
        subprocess.run([PY, "scripts/evidential_make_configs.py", "ensemble", "--arm", arm, "--fold", fold,
                        "--seed", str(seed), "--lam", str(lam), "--anneal", str(anneal)], check=True, cwd=REPO_ROOT)
        name = f"ens_{arm}_{fold}_s{seed}" + (f"_{tag}" if arm == "E" else "") + ".yaml"
        plans.append((arm, fold, seed, f"configs/evidential/{name}"))
    plans.append(("C", "fold1", 42, "configs/region_ensemble_deepdropout_fold1.yaml"))  # already trained

    jobs = []
    for arm, fold, seed, cfg in plans:
        mc = "20" if arm == "C" else "0"
        jobs.append((arm, cfg, f"grasp_{fold}", mc, f"{arm}_grasp_{fold}_s{seed}"))
        if fold == "fold1" and seed == 42:
            for target, zipname in (("endovis2018", "endovis2018.zip"), ("endovis2017", "endovis2017.zip")):
                jobs.append((arm, cfg, target, mc, f"{arm}_{target}_fold1_s42", str(Path.home() / "Desktop/endovis_data" / zipname)))
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)

    def worker(gpu: int) -> None:
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            arm, cfg, target, mc, name = j[:5]
            out = OUT / "extract" / f"{name}.npz"
            if out.exists():
                continue
            cmd = [PY, "scripts/evidential_extract.py", "--ensemble-config", cfg, "--target", target, "--data-root", "GraSP",
                   "--mc-samples", mc, "--device", f"cuda:{gpu}", "--out", str(out)]
            if len(j) > 5:
                cmd += ["--zip", j[5]]
            with open(OUT / "extract" / f"{name}.log", "w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            print("extracted", name, flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    {"choose": choose, "train": train, "extract": extract}[sys.argv[1]]()
