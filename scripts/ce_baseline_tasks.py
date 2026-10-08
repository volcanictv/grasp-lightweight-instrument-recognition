"""Task list for scripts/campaign.py on titanxp: train the 12 cross-entropy baseline members (arm C official, scripts/make_ce_official_configs.py), seed 42 first.

    export GRASP_DATA_ROOT=...; python scripts/campaign.py --out ~/ce_campaign --gpus 0,1 --budget-hours 12 --tasks-module ce_baseline_tasks
Each task trains one member with scripts/train.py (fixed schedule, last epoch) and writes <out>/<member>_s<seed>.ok with the run directory.
"""
from __future__ import annotations

import os
from pathlib import Path

MEMBERS = {"resnet50_320": 50, "resnet50_224": 30, "baseline": 15, "letterbox_crop": 15}   # estimated minutes on a Titan Xp
PY = os.environ.get("PY_MAIN", "/home/yzx/miniconda3/envs/surgical/bin/python")


def build(cfg: dict, Task) -> list:
    out, code = Path(cfg["out"]), cfg["code"]
    data = os.environ["GRASP_DATA_ROOT"]
    tasks = []
    for seed, prio in ((42, 9), (43, 6), (44, 3)):
        for m, minutes in MEMBERS.items():
            stem = f"armC_{m}_official_s{seed}"
            marker = out / f"{m}_s{seed}.ok"
            cmd = (f"cd {code} && export GRASP_DATA_ROOT='{data}' PYTHONUNBUFFERED=1 && "
                   f"{PY} -u scripts/train.py configs/arms/{stem}.yaml --data-root '{data}' --device cuda:0 --num-workers 3 --experiments-dir experiments && "
                   f"d=$(ls -d experiments/{stem}_2* | tail -n 1) && test -f $d/best.pt && echo $d > {marker}")
            tasks.append(Task(f"{m}_s{seed}", cmd, outputs=[str(marker)], minutes=minutes, priority=prio + (1 if m == 'resnet50_320' else 0)))
    return tasks
