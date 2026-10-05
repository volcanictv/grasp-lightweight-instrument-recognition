"""Rung C of the real-time addendum (docs/reports/gtbox_sam_protocol.md, 2026-10-05 second), development part on fold1.

Past evidence without a tracker: an instrument's single-frame belief is fused with the beliefs of earlier instruments that overlap its box in the previous
k keyframes of the same case. No ground-truth identity is used for the fusion (the labels only measure how pure the matches are). Fusion is the
evidential frame fusion of the pipeline (sum over frames of max(mu_f) * mu_f) and is applied to the gated instruments only (top 20% by epistemic score S1).

Members are the arm N fold1 members (trained on fold2; three members, weights 0.40 / 0.30 / 0.30) with ground-truth masks. Settings: overlap threshold in
{0.3, 0.5}, k in {3, 5, 10}; rule: highest mean fold1 accuracy over the three seeds, then the cheapest (smallest k, then highest threshold) within 0.001.

Usage (titanxp, surgical environment):
    python scripts/rt_assoc_fold1.py --out docs/reports/realtime/assoc_fold1.json
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import crop_from_box
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.models import build_model

WEIGHTS = {"resnet50_320": 0.40, "baseline": 0.30, "letterbox_crop": 0.30}
SEEDS = (42, 43, 44)
THRESHOLDS = (0.3, 0.5)
KS = (3, 5, 10)
GATE_SHARE = 0.20
TOLERANCE = 0.001


def decode_gt(seg, height: int, width: int) -> np.ndarray:
    rle = seg
    if isinstance(rle.get("counts"), list):
        rle = mask_codec.frPyObjects(rle, height, width)
    return mask_codec.decode(rle).astype(bool)


def extract(ds: GraspRegionDataset, config: Path, device: str, cache: Path) -> dict[str, np.ndarray]:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    cfg = yaml.safe_load(config.read_text())
    members = []
    for m in cfg["members"]:
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        members.append((m["label"], net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))
    by_frame = defaultdict(list)
    for idx, (file_name, *_rest) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    out = {label: np.zeros((len(ds.instances), 7), dtype=np.float32) for label, *_ in members}
    for n, (file_name, indices) in enumerate(sorted(by_frame.items())):
        frame = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        for idx in indices:
            _f, seg, box, _l = ds.instances[idx]
            mask = decode_gt(seg, frame.shape[0], frame.shape[1])
            for label, net, transform, letterbox in members:
                crop = crop_from_box(frame, mask, box, letterbox)
                with torch.no_grad():
                    out[label][idx] = net(transform(Image.fromarray(crop)).unsqueeze(0).to(device)).float().cpu().numpy()[0]
        if (n + 1) % 200 == 0:
            print(f"{n + 1}/{len(by_frame)} frames", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "experiments" / "rt_assoc")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = GraspRegionDataset(args.data_root, "fold1", letterbox=True)
    y = np.array([inst[3] for inst in ds.instances])
    boxes = [inst[2] for inst in ds.instances]
    case = np.array([inst[0].split("/")[0] for inst in ds.instances])
    frame_no = np.array([int(Path(inst[0]).stem) for inst in ds.instances])

    # keyframe order within each case
    frames_of_case: dict[str, list[int]] = defaultdict(list)
    for c, f in sorted(set(zip(case.tolist(), frame_no.tolist()))):
        frames_of_case[c].append(f)
    gaps = [b - a for fs in frames_of_case.values() for a, b in zip(fs, fs[1:])]
    members_at: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (c, f) in enumerate(zip(case.tolist(), frame_no.tolist())):
        members_at[(c, f)].append(i)
    pos_in_case = {(c, f): k for c, fs in frames_of_case.items() for k, f in enumerate(fs)}

    # matches[(thr, k)] = list over instances of earlier instrument indices
    matches: dict[tuple[float, int], list[list[int]]] = {}
    for thr in THRESHOLDS:
        for k in KS:
            m = []
            for i in range(len(y)):
                c, f = case[i], int(frame_no[i])
                p = pos_in_case[(c, f)]
                found = []
                for q in range(max(0, p - k), p):
                    cands = members_at[(c, frames_of_case[c][q])]
                    best, best_iou = None, thr
                    for j in cands:
                        v = iou_xywh(boxes[i], boxes[j])
                        if v >= best_iou:
                            best, best_iou = j, v
                    if best is not None:
                        found.append(best)
                m.append(found)
            matches[(thr, k)] = m

    stats = {"keyframe_gap_median": float(np.median(gaps)), "n_instances": int(len(y))}
    for (thr, k), m in matches.items():
        has = np.array([len(f) > 0 for f in m])
        pure = [np.mean([y[j] == y[i] for j in f]) for i, f in enumerate(m) if f]
        stats[f"thr{thr}_k{k}"] = {"share_with_a_match": float(has.mean()), "mean_matches": float(np.mean([len(f) for f in m])),
                                   "match_label_purity": float(np.mean(pure)) if pure else None}

    per_seed = []
    for seed in SEEDS:
        cfg = REPO_ROOT / "configs" / "arms" / f"ens_N_fold1_s{seed}.yaml"
        logits = extract(ds, cfg, args.device, args.cache_dir / f"armN_fold1_s{seed}.npz")
        alpha = sum(w * alpha_from_logits(logits[k]) for k, w in WEIGHTS.items())
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        s1 = variance_scores(alpha)["epistemic"]
        base = mu.argmax(axis=1)
        gate = s1 >= np.quantile(s1, 1.0 - GATE_SHARE)
        rows = {"base_accuracy": float((base == y).mean())}
        for (thr, k), m in matches.items():
            for gated in (True, False):
                pred = base.copy()
                for i in range(len(y)):
                    if (gated and not gate[i]) or not m[i]:
                        continue
                    ids = [i] + m[i]
                    pred[i] = int((mu[ids].max(axis=1)[:, None] * mu[ids]).sum(axis=0).argmax())
                rows[f"thr{thr}_k{k}_{'gated' if gated else 'all'}"] = float((pred == y).mean())
        per_seed.append(rows)
        print(seed, rows, flush=True)

    keys = [k for k in per_seed[0] if k.endswith("_gated")]
    mean = {k: float(np.mean([r[k] for r in per_seed])) for k in keys}
    mean["base_accuracy"] = float(np.mean([r["base_accuracy"] for r in per_seed]))
    best = max(mean[k] for k in keys)
    near = [k for k in keys if mean[k] >= best - TOLERANCE]
    near.sort(key=lambda s: (int(s.split("_k")[1].split("_")[0]), -float(s.split("thr")[1].split("_")[0])))
    chosen = near[0]
    out = {"stats": stats, "per_seed": per_seed, "mean_gated": mean, "chosen": chosen, "chosen_accuracy": mean[chosen],
           "gain_over_base": mean[chosen] - mean["base_accuracy"], "gate_share": GATE_SHARE}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("chosen", "chosen_accuracy", "gain_over_base")}, indent=1))
    print("mean base", mean["base_accuracy"])


if __name__ == "__main__":
    main()
