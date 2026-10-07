"""The task list of the campaign (scripts/campaign.py runs it). Every task is a shell command that sees one GPU as cuda:0.

Environment (all have defaults for the cluster layout under $WORK, default ~/grasp_work; on titanxp set them to the local environments):
  GRASP_DATA_ROOT   the GraSP folder (annotations/ and frames-001/); the registry written by scripts/build_cv_splits.py must be in annotations/
  PY_MAIN           python with torch, SAM2, the classifier code          PY_SAM3  python with transformers 5.x (default PY_MAIN)       PY_ET  python of the EdgeTAM environment
  SAM2_CKPT_DIR     SAM2.1 checkpoints                                     EDGETAM_CKPT  the EdgeTAM checkpoint
  NEIGHBOURS        colon-separated neighbour-crop folders of the official training cases (default: the two under experiments/temporal_neighbors)

Per cross-validation fold (cv5_f0 .. cv5_f4, held-out cases never seen in training or selection) the chain is
  SAM2 training, SAM3 training  ->  masks of the held-out cases  ->  four classifier members (needs the neighbour crops of every training case)  ->  ensemble config  ->  single-pass logits
  ->  choose the instruments to track (highest S1)  ->  SAM2-large tracking over +-10 frames  ->  scoring
With --smoke the same chain runs on the tiny splits (cv5_smoke_*) with one epoch and a handful of instruments, to test every command and path before any long run.
"""
from __future__ import annotations

import os
from pathlib import Path

MEMBERS = {"resnet50_320": (35, 7), "resnet50_224": (20, 6), "baseline": (12, 5), "letterbox_crop": (12, 5)}  # member: (A100 minutes, priority)


def build(cfg: dict, Task) -> list:
    out, code, smoke = Path(cfg["out"]), cfg["code"], cfg["smoke"]
    work = Path(os.environ.get("WORK", Path.home() / "grasp_work"))
    data = os.environ.get("GRASP_DATA_ROOT")
    if not data:
        raise SystemExit("set GRASP_DATA_ROOT to the GraSP folder")
    py = os.environ.get("PY_MAIN", str(work / "envs" / "main" / "bin" / "python"))
    py3 = os.environ.get("PY_SAM3", py)
    pyet = os.environ.get("PY_ET", str(work / "envs" / "edgetam" / "bin" / "python"))
    ck = os.environ.get("SAM2_CKPT_DIR", str(work / "checkpoints" / "sam2"))
    etck = os.environ.get("EDGETAM_CKPT", str(work / "EdgeTAM" / "checkpoints" / "edgetam.pt"))
    registry = f"{data}/annotations/cv5_splits.json"
    tmp = f"/tmp/{os.environ.get('USER', 'user')}_campaign"
    pre = f"cd {code} && export GRASP_DATA_ROOT='{data}' GRASP_EXTRA_SPLITS='{registry}' PYTHONUNBUFFERED=1 && "
    nb_official = os.environ.get("NEIGHBOURS", f"{code}/experiments/temporal_neighbors/fold1:{code}/experiments/temporal_neighbors/fold2").split(":")
    nb_test = f"{out}/neighbours/test"
    neighbours = " ".join([*nb_official, nb_test])
    t = lambda minutes, full: max(2.0, minutes / 10) if smoke else full  # estimate in minutes: the smoke runs are short

    tasks = [Task("neighbours_test", f"{pre}{pyet} -u scripts/build_temporal_neighbors.py --json-split test --out-dir {nb_test} --sam-checkpoint {etck} --data-root {data} "
                  f"{'--max-frames 3 ' if smoke else ''}--device cuda:0", outputs=[f"{nb_test}/meta.json"], minutes=t(0, 60), priority=10)]
    folds = ["cv5_smoke"] if smoke else [f"cv5_f{k}" for k in range(5)]
    for fold in folds:
        fo = f"{out}/{fold}"
        lim = "--limit 12 " if smoke else ""
        tasks += [
            Task(f"{fold}_sam2", f"{pre}{py} -u scripts/finetune_sam2_gtbox.py --train-split {fold}_train --dev-split {fold}_dev --fixed-epochs {1 if smoke else 5} --unfreeze-blocks 4 {lim}"
                 f"--sam-checkpoint {ck}/sam2.1_hiera_large.pt --data-root {data} --out-dir {fo}/sam2 --device cuda:0", outputs=[f"{fo}/sam2/weights.pt"], minutes=t(0, 90), priority=8),
            Task(f"{fold}_sam3", f"{pre}{py3} -u scripts/finetune_sam3_gtbox.py --train-split {fold}_train --dev-split {fold}_dev --epochs {1 if smoke else 3} --unfreeze-layers 4 --save-last "
                 f"--dev-stride {1 if smoke else 4} --final-stride {1 if smoke else 8} {lim}--data-root {data} --out-dir {fo}/sam3 --device cuda:0", outputs=[f"{fo}/sam3/weights.pt"],
                 minutes=t(0, 75), priority=8),
            Task(f"{fold}_masks", f"{pre}{py3} -u scripts/gtbox_sam_masks_ensemble.py --split {fold}_test --sam2-weights {fo}/sam2/weights.pt --sam3-weights {fo}/sam3/weights.pt "
                 f"--sam2-checkpoint {ck}/sam2.1_hiera_large.pt {'--limit-frames 10 ' if smoke else ''}--data-root {data} --out {fo}/masks_part0.pkl --device cuda:0 && "
                 f"{py3} scripts/merge_gtbox_mask_parts.py --parts {fo}/masks_part0.pkl --variant {fold} --out-dir {fo}/final", deps=[f"{fold}_sam2", f"{fold}_sam3"],
                 outputs=[f"{fo}/final/masks.pkl"], minutes=t(0, 25), priority=6),
        ]
        for m, (minutes, prio) in MEMBERS.items():
            tasks.append(Task(f"{fold}_clf_{m}", f"{pre}{py} -u scripts/cv_pipeline.py clf-run --fold {fold} --member {m} --seed 42 --dir {fo}/clf --neighbours {neighbours} "
                              f"--data-root {data} --epochs {1 if smoke else 20} --python {py}", deps=["neighbours_test"], outputs=[f"{fo}/clf/{m}.ok"], minutes=t(minutes, minutes), priority=prio))
        tasks += [
            Task(f"{fold}_ens", f"{pre}{py} scripts/cv_pipeline.py ens-config --dir {fo}/clf --out {fo}/ens4.yaml", needs_gpu=False, deps=[f"{fold}_clf_{m}" for m in MEMBERS],
                 outputs=[f"{fo}/ens4.yaml"], minutes=1, priority=9),
            Task(f"{fold}_logits", f"{pre}{py} -u scripts/extract_logits_from_masks.py --masks {fo}/final/masks.pkl --ensemble-config {fo}/ens4.yaml --split {fold}_test --data-root {data} "
                 f"--device cuda:0 --out {fo}/final/logits.npz", deps=[f"{fold}_masks", f"{fold}_ens"], outputs=[f"{fo}/final/logits.npz"], minutes=t(0, 10), priority=6),
            Task(f"{fold}_select", f"{pre}{py} scripts/cv_pipeline.py select --logits {fo}/final/logits.npz --frac 0.5 {'--max-instruments 6 ' if smoke else ''}--out {fo}/final/track_idx.json",
                 needs_gpu=False, deps=[f"{fold}_logits"], outputs=[f"{fo}/final/track_idx.json"], minutes=1, priority=6),
            Task(f"{fold}_track", f"{pre}{py} -u scripts/evaluate_temporal_track_ensemble.py --split {fold}_test --window 10 --ensemble-config {fo}/ens4.yaml "
                 f"--sam2-checkpoint {ck}/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --init-masks {fo}/final/masks.pkl "
                 f"--error-cases-json {fo}/final/track_idx.json --tmp-dir {tmp}/{fold} --frame-logits-out {fo}/final/tracked/frames_shard0.npz --masks-out {fo}/final/tracked/masks_shard0.pkl "
                 f"--device cuda:0 --save-every 25 --data-root {data} --out {fo}/final/tracked/tracked_shard0.json", deps=[f"{fold}_select"], outputs=[f"{fo}/final/tracked/frames_shard0.npz"],
                 minutes=t(0, 100), priority=4),
            Task(f"{fold}_score", f"{pre}{py} scripts/cv_pipeline.py score --split {fold}_test --masks {fo}/final/masks.pkl --logits {fo}/final/logits.npz --tracked-dir {fo}/final/tracked "
                 f"--frames-out {fo}/final/frames.pkl --out {fo}/score.json --python {py}", needs_gpu=False, deps=[f"{fold}_track"], outputs=[f"{fo}/score.json"], minutes=10, priority=1),
        ]
    return tasks
