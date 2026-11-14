"""Task list of the ground-truth-mask run of the 5-fold cross-validation (scripts/campaign.py --tasks-module campaign_tasks_gtmask).

The classifier members, ensemble configs and held-out cases of the finished campaign (scripts/campaign_tasks.py, directory $OLD_CAMPAIGN, default ~/grasp_work/campaign_cv5) are reused as they are:
nothing is retrained. Per fold the chain is: annotated masks of the held-out cases -> single-pass logits of the existing members on them -> the 50% most uncertain instruments (S1) ->
SAM2-large tracking over +-10 frames from the annotated masks -> scoring with the same cv_pipeline.py score as the SAM-mask run, so the two are comparable cell by cell.
Environment as in scripts/campaign_tasks.py, plus OLD_CAMPAIGN. With --smoke the chain runs on the tiny cv5_smoke split of a smoke campaign directory.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path


def build(cfg: dict, Task) -> list:
    out, code, smoke = Path(cfg["out"]), cfg["code"], cfg["smoke"]
    work = Path(os.environ.get("WORK", Path.home() / "grasp_work"))
    data = os.environ.get("GRASP_DATA_ROOT")
    if not data:
        raise SystemExit("set GRASP_DATA_ROOT to the GraSP folder")
    old = Path(os.environ.get("OLD_CAMPAIGN", work / "campaign_cv5"))
    py = os.environ.get("PY_MAIN", str(work / "envs" / "main" / "bin" / "python"))
    ck = os.environ.get("SAM2_CKPT_DIR", str(work / "checkpoints" / "sam2"))
    dq = shlex.quote(data)
    registry = f"{data}/annotations/cv5_splits.json"
    tmp = f"/tmp/{os.environ.get('USER', 'user')}_campaign_gtmask"
    pre = f"cd {code} && export GRASP_DATA_ROOT='{data}' GRASP_EXTRA_SPLITS='{registry}' PYTHONUNBUFFERED=1 && "
    shards = max(1, int(os.environ.get("TRACK_SHARDS", "2")))
    tasks = []
    for fold in (["cv5_smoke"] if smoke else [f"cv5_f{k}" for k in range(5)]):
        fo, ens = f"{out}/{fold}", f"{old}/{fold}/ens4.yaml"
        limit = "--max-instruments 6 " if smoke else ""
        tasks += [
            Task(f"{fold}_gtmasks", f"{pre}{py} scripts/gt_masks_to_pkl.py --split {fold}_test --data-root {dq} --out {fo}/final/masks.pkl", needs_gpu=False,
                 outputs=[f"{fo}/final/masks.pkl"], minutes=2, priority=9),
            Task(f"{fold}_logits", f"{pre}{py} -u scripts/extract_logits_from_masks.py --masks {fo}/final/masks.pkl --ensemble-config {ens} --split {fold}_test --data-root {dq} "
                 f"--device cuda:0 --out {fo}/final/logits.npz", deps=[f"{fold}_gtmasks"], outputs=[f"{fo}/final/logits.npz"], minutes=10, priority=8, light=True, mem_gb=6),
            Task(f"{fold}_select", f"{pre}{py} scripts/cv_pipeline.py select --logits {fo}/final/logits.npz --frac 0.5 {limit}--shards {shards} --out {fo}/final/track_idx.json",
                 needs_gpu=False, deps=[f"{fold}_logits"], outputs=[f"{fo}/final/track_idx.json"], minutes=1, priority=8),
        ]
        names = []
        for k in range(shards):
            idx = f"{fo}/final/track_idx_shard{k}.json" if shards > 1 else f"{fo}/final/track_idx.json"
            name = f"{fold}_track" if shards == 1 else f"{fold}_track_sh{k}"
            names.append(name)
            tasks.append(Task(name, f"{pre}{py} -u scripts/evaluate_temporal_track_ensemble.py --split {fold}_test --window 10 --ensemble-config {ens} "
                              f"--sam2-checkpoint {ck}/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --init-masks {fo}/final/masks.pkl "
                              f"--error-cases-json {idx} --tmp-dir {tmp}/{fold}_{k} --frame-logits-out {fo}/final/tracked/frames_shard{k}.npz --masks-out {fo}/final/tracked/masks_shard{k}.pkl "
                              f"--device cuda:0 --save-every 25 --data-root {dq} --out {fo}/final/tracked/tracked_shard{k}.json",
                              deps=[f"{fold}_select"], outputs=[f"{fo}/final/tracked/frames_shard{k}.npz"], minutes=(5 if smoke else 100) / shards, priority=4, light=True, mem_gb=12))
        tasks.append(Task(f"{fold}_score", f"{pre}{py} scripts/cv_pipeline.py score --split {fold}_test --masks {fo}/final/masks.pkl --logits {fo}/final/logits.npz --tracked-dir {fo}/final/tracked "
                          f"--frames-out {fo}/final/frames.pkl --out {fo}/score.json --python {py}", needs_gpu=False, deps=names, outputs=[f"{fo}/score.json"], minutes=10, priority=1))
    return tasks
