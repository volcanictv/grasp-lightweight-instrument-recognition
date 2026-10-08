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
import shlex
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
    dq = shlex.quote(data)
    registry = f"{data}/annotations/cv5_splits.json"
    tmp = f"/tmp/{os.environ.get('USER', 'user')}_campaign"
    pre = f"cd {code} && export GRASP_DATA_ROOT='{data}' GRASP_EXTRA_SPLITS='{registry}' PYTHONUNBUFFERED=1 && "
    nb_official = os.environ.get("NEIGHBOURS", f"{code}/experiments/temporal_neighbors/fold1:{code}/experiments/temporal_neighbors/fold2").split(":")
    nb_test = f"{out}/neighbours/test"
    neighbours = " ".join([*nb_official, nb_test])
    t = lambda minutes, full: max(2.0, minutes / 10) if smoke else full  # estimate in minutes: the smoke runs are short
    shards = max(1, int(os.environ.get("TRACK_SHARDS", "2")))   # tracking of a fold is split in this many tasks, so a fold's chain is shorter and the last GPUs are not left waiting
    workers = int(os.environ.get("CAMPAIGN_WORKERS", "4"))      # data-loader workers of a classifier run: the CPUs per concurrently running GPU task

    def track(name, fold, window, ens, masks, idx_dir, tdir, tmpname, minutes, prio, dep):
        """The tracking tasks of one run: one per shard of the chosen instruments (cv_pipeline.py select writes track_idx_shard<k>.json)."""
        out_tasks = []
        for k in range(shards):
            idx = f"{idx_dir}/track_idx_shard{k}.json" if shards > 1 else f"{idx_dir}/track_idx.json"
            out_tasks.append(Task(name if shards == 1 else f"{name}_sh{k}",
                                  f"{pre}{py} -u scripts/evaluate_temporal_track_ensemble.py --split {fold}_test --window {window} --ensemble-config {ens} "
                                  f"--sam2-checkpoint {ck}/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --init-masks {masks} "
                                  f"--error-cases-json {idx} --tmp-dir {tmp}/{tmpname}_{k} --frame-logits-out {tdir}/frames_shard{k}.npz --masks-out {tdir}/masks_shard{k}.pkl "
                                  f"--device cuda:0 --save-every 25 --data-root {dq} --out {tdir}/tracked_shard{k}.json",
                                  deps=[dep], outputs=[f"{tdir}/frames_shard{k}.npz"], minutes=minutes / shards, priority=prio))
        return out_tasks, [x.name for x in out_tasks]

    tasks = [Task("neighbours_test", f"{pre}{pyet} -u scripts/build_temporal_neighbors.py --json-split test --out-dir {nb_test} --sam-checkpoint {etck} --data-root {dq} "
                  f"{'--max-frames 3 ' if smoke else ''}--device cuda:0", outputs=[f"{nb_test}/meta.json"], minutes=t(0, 60), priority=10)]
    folds = ["cv5_smoke"] if smoke else [f"cv5_f{k}" for k in range(5)]
    for fold in folds:
        fo = f"{out}/{fold}"
        lim = "--limit 12 " if smoke else ""
        tasks += [
            Task(f"{fold}_sam2", f"{pre}{py} -u scripts/finetune_sam2_gtbox.py --train-split {fold}_train --dev-split {fold}_dev --fixed-epochs {1 if smoke else 5} --unfreeze-blocks 4 {lim}"
                 f"--sam-checkpoint {ck}/sam2.1_hiera_large.pt --data-root {dq} --out-dir {fo}/sam2 --device cuda:0", outputs=[f"{fo}/sam2/weights.pt"], minutes=t(0, 90), priority=8),
            Task(f"{fold}_sam3", f"{pre}{py3} -u scripts/finetune_sam3_gtbox.py --train-split {fold}_train --dev-split {fold}_dev --epochs {1 if smoke else 3} --unfreeze-layers 4 --save-last "
                 f"--dev-stride {1 if smoke else 4} --final-stride {1 if smoke else 8} {lim}--data-root {dq} --out-dir {fo}/sam3 --device cuda:0", outputs=[f"{fo}/sam3/weights.pt"],
                 minutes=t(0, 75), priority=8),
            Task(f"{fold}_masks", f"{pre}{py3} -u scripts/gtbox_sam_masks_ensemble.py --split {fold}_test --sam2-weights {fo}/sam2/weights.pt --sam3-weights {fo}/sam3/weights.pt "
                 f"--sam2-checkpoint {ck}/sam2.1_hiera_large.pt {'--limit-frames 10 ' if smoke else ''}--data-root {dq} --out {fo}/masks_part0.pkl --device cuda:0 && "
                 f"{py3} scripts/merge_gtbox_mask_parts.py --parts {fo}/masks_part0.pkl --variant {fold} --out-dir {fo}/final", deps=[f"{fold}_sam2", f"{fold}_sam3"],
                 outputs=[f"{fo}/final/masks.pkl"], minutes=t(0, 25), priority=6),
        ]
        for m, (minutes, prio) in MEMBERS.items():
            tasks.append(Task(f"{fold}_clf_{m}", f"{pre}{py} -u scripts/cv_pipeline.py clf-run --fold {fold} --member {m} --seed 42 --dir {fo}/clf --neighbours {neighbours} "
                              f"--data-root {dq} --epochs {1 if smoke else 20} --workers {workers} --python {py}", deps=["neighbours_test"], outputs=[f"{fo}/clf/{m}.ok"],
                              minutes=t(minutes, minutes), priority=prio, light=True))
        tasks += [
            Task(f"{fold}_ens", f"{pre}{py} scripts/cv_pipeline.py ens-config --dir {fo}/clf --out {fo}/ens4.yaml", needs_gpu=False, deps=[f"{fold}_clf_{m}" for m in MEMBERS],
                 outputs=[f"{fo}/ens4.yaml"], minutes=1, priority=9),
            Task(f"{fold}_logits", f"{pre}{py} -u scripts/extract_logits_from_masks.py --masks {fo}/final/masks.pkl --ensemble-config {fo}/ens4.yaml --split {fold}_test --data-root {dq} "
                 f"--device cuda:0 --out {fo}/final/logits.npz", deps=[f"{fold}_masks", f"{fold}_ens"], outputs=[f"{fo}/final/logits.npz"], minutes=t(0, 10), priority=6, light=True),
            Task(f"{fold}_select", f"{pre}{py} scripts/cv_pipeline.py select --logits {fo}/final/logits.npz --frac 0.5 {'--max-instruments 6 ' if smoke else ''}--shards {shards} --out {fo}/final/track_idx.json",
                 needs_gpu=False, deps=[f"{fold}_logits"], outputs=[f"{fo}/final/track_idx.json"], minutes=1, priority=6),
        ]
        trk, trk_names = track(f"{fold}_track", fold, 10, f"{fo}/ens4.yaml", f"{fo}/final/masks.pkl", f"{fo}/final", f"{fo}/final/tracked", fold, t(0, 100), 4, f"{fold}_select")
        tasks += trk + [
            Task(f"{fold}_score", f"{pre}{py} scripts/cv_pipeline.py score --split {fold}_test --masks {fo}/final/masks.pkl --logits {fo}/final/logits.npz --tracked-dir {fo}/final/tracked "
                 f"--frames-out {fo}/final/frames.pkl --out {fo}/score.json --python {py}", needs_gpu=False, deps=trk_names, outputs=[f"{fo}/score.json"], minutes=10, priority=1),
        ]
        if smoke:
            continue
        # extra 1: the tracking window (k frames each side) on the same masks, logits and instruments; the +-10 run above is the main one
        for k in (2, 3, 5):
            wtrk, wnames = track(f"{fold}_track_w{k}", fold, k, f"{fo}/ens4.yaml", f"{fo}/final/masks.pkl", f"{fo}/final", f"{fo}/final/tracked_w{k}", f"{fold}_w{k}", 25 + 5 * k, 2, f"{fold}_select")
            tasks += wtrk + [
                Task(f"{fold}_score_w{k}", f"{pre}{py} scripts/cv_pipeline.py score --split {fold}_test --masks {fo}/final/masks.pkl --logits {fo}/final/logits.npz "
                     f"--tracked-dir {fo}/final/tracked_w{k} --frames-out {fo}/final/frames_w{k}.pkl --out {fo}/score_w{k}.json --python {py}", needs_gpu=False,
                     deps=wnames, outputs=[f"{fo}/score_w{k}.json"], minutes=10, priority=1),
            ]
    # extra 2: variance from the segmenter's training seed, on fold 0 only: a second and third SAM2 fine-tune (SAM3 and the classifiers are reused), then the same chain downstream
    fold, fo = "cv5_f0", f"{out}/cv5_f0"
    for s in () if smoke else (1, 2):
        so = f"{out}/cv5_f0_seed{s}"
        tasks += [
            Task(f"seed{s}_sam2", f"{pre}{py} -u scripts/finetune_sam2_gtbox.py --train-split {fold}_train --dev-split {fold}_dev --fixed-epochs 5 --unfreeze-blocks 4 --seed {s} "
                 f"--sam-checkpoint {ck}/sam2.1_hiera_large.pt --data-root {dq} --out-dir {so}/sam2 --device cuda:0", outputs=[f"{so}/sam2/weights.pt"], minutes=90, priority=3),
            Task(f"seed{s}_masks", f"{pre}{py3} -u scripts/gtbox_sam_masks_ensemble.py --split {fold}_test --sam2-weights {so}/sam2/weights.pt --sam3-weights {fo}/sam3/weights.pt "
                 f"--sam2-checkpoint {ck}/sam2.1_hiera_large.pt --data-root {dq} --out {so}/masks_part0.pkl --device cuda:0 && "
                 f"{py3} scripts/merge_gtbox_mask_parts.py --parts {so}/masks_part0.pkl --variant seed{s} --out-dir {so}/final", deps=[f"seed{s}_sam2", f"{fold}_sam3"],
                 outputs=[f"{so}/final/masks.pkl"], minutes=25, priority=3),
            Task(f"seed{s}_logits", f"{pre}{py} -u scripts/extract_logits_from_masks.py --masks {so}/final/masks.pkl --ensemble-config {fo}/ens4.yaml --split {fold}_test --data-root {dq} "
                 f"--device cuda:0 --out {so}/final/logits.npz", deps=[f"seed{s}_masks", f"{fold}_ens"], outputs=[f"{so}/final/logits.npz"], minutes=10, priority=3),
            Task(f"seed{s}_select", f"{pre}{py} scripts/cv_pipeline.py select --logits {so}/final/logits.npz --frac 0.5 --shards {shards} --out {so}/final/track_idx.json", needs_gpu=False,
                 deps=[f"seed{s}_logits"], outputs=[f"{so}/final/track_idx.json"], minutes=1, priority=3),
        ]
        strk, snames = track(f"seed{s}_track", fold, 10, f"{fo}/ens4.yaml", f"{so}/final/masks.pkl", f"{so}/final", f"{so}/final/tracked", f"seed{s}", 100, 2, f"seed{s}_select")
        tasks += strk + [
            Task(f"seed{s}_score", f"{pre}{py} scripts/cv_pipeline.py score --split {fold}_test --masks {so}/final/masks.pkl --logits {so}/final/logits.npz --tracked-dir {so}/final/tracked "
                 f"--frames-out {so}/final/frames.pkl --out {so}/score.json --python {py}", needs_gpu=False, deps=snames, outputs=[f"{so}/score.json"], minutes=10, priority=1),
        ]
    return tasks
