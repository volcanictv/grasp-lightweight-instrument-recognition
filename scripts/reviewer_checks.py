"""Four CPU checks of the paper's claims, from saved logits (no new model runs), each reported whatever the result:
  weights     the final pipeline (SAM2 + SAM3 masks, tracked over +-10 frames, 833 refined) with the registered member weights (0.40, 0.20, 0.20, 0.20) and with equal weights 0.25; also the share of test
              frames whose instruments are all right after the single pass and after refinement, mean of three seeds
  window      the same pipeline fusing only the frames within k of the keyframe, k = 0, 1, 2, 3, 5, 10 (how many neighbouring frames the gain needs)
  unanimous   share of the errors on which every member of an ensemble predicts the same wrong class that fall in the 20% highest-scoring instruments, per method (held-out fold)
Usage (titanxp, surgical environment, repo root): python scripts/reviewer_checks.py weights|window|unanimous --out docs/reports/reviewer_checks_<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from pycocotools import mask as mask_codec
from scipy.special import softmax

from evidential_analyze import LABELS, arm_c, arm_e, load
from evidential_seeds_e2e_eval import CONFIGS, ORDER
from gtbox_sam_final_eval import load_tracked
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint

SEEDS = (42, 43, 44)
W_REG = CONFIGS["four"]
W_FLAT = {k: 0.25 for k in ORDER}
D = REPO_ROOT / "experiments" / "gtbox_sam" / "final"


def alpha_of(logits: np.ndarray, W: dict) -> np.ndarray:
    return sum(W[k] * alpha_from_logits(logits[..., ORDER.index(k), :]) for k in ORDER)


def fuse(alpha_frames: np.ndarray) -> tuple[int, float]:
    mu = alpha_frames / alpha_frames.sum(axis=1, keepdims=True)
    c = ((mu.max(axis=1)[:, None]) * mu).sum(axis=0)
    return int(c.argmax()), float(c.max() / c.sum())


def tracked_dir(seed: int) -> Path:
    return D / ("tracked" if seed == 42 else f"tracked_s{seed}")


class Test:
    def __init__(self) -> None:
        root = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))
        self.ds = GraspRegionDataset(root, "test", letterbox=True)
        self.masks = pickle.loads((D / "masks.pkl").read_bytes())
        self.sam = {i: mask_codec.decode(r).astype(bool) for i, r in self.masks.items()}
        self.by_frame: dict[str, list[int]] = defaultdict(list)
        for i, inst in enumerate(self.ds.instances):
            self.by_frame[inst[0]].append(i)
        self.z = {s: np.load(D / f"logits_s{s}.npz") for s in SEEDS}
        self.y, self.done = self.z[42]["y"], self.z[42]["done"]
        self.tracked = {s: load_tracked(tracked_dir(s)) for s in SEEDS}
        self.offsets = None

    def score(self, pred: np.ndarray, conf: np.ndarray) -> dict:
        frames, right = [], 0
        n_frames = 0
        for fn, idx in self.by_frame.items():
            if any(i not in self.masks for i in idx):
                continue
            shape = tuple(self.ds.instances[idx[0]][1]["size"])
            gt = paint(shape, [(decode_instance_mask(self.ds.instances[i][1]).astype(bool), int(self.y[i]) + 1, 0.0) for i in idx])
            frames.append(frame_class_ious(paint(shape, [(self.sam[i], int(pred[i]) + 1, float(conf[i])) for i in idx if i in self.sam and self.done[i]]), gt))
            n_frames += 1
            right += all(pred[i] == self.y[i] for i in idx if self.done[i])
        out = aggregate(frames)
        return {k: float(out[k]) for k in ("mIoU", "IoU", "mcIoU")} | {"frames_all_right": right / max(1, n_frames), "instance_accuracy": float((pred[self.done] == self.y[self.done]).mean())}

    def run(self, seed: int, W: dict, budget: int, rows=None) -> dict:
        alpha = alpha_of(np.stack([self.z[seed]["det_" + k] for k in ORDER], axis=1), W)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        s1 = variance_scores(alpha)["epistemic"]
        flag = np.zeros(len(self.y), bool)
        flag[np.argsort(-s1, kind="stable")[:budget]] = True
        pred, conf = mu.argmax(axis=1).copy(), mu.max(axis=1).copy()
        base_pred = pred.copy()
        missing = 0
        for i in np.where(flag & self.done)[0]:
            fr = self.tracked[seed].get(int(i))
            if fr is None:
                missing += 1
                continue
            if rows is not None:
                keep = rows(int(i), fr)
                if keep is None:
                    missing += 1
                    continue
                fr = fr[keep]
            pred[i], conf[i] = fuse(alpha_of(fr, W))
        return {**self.score(pred, conf), "missing": missing, "single": self.score(base_pred, mu.max(axis=1))}


def load_offsets() -> dict:
    out = {}
    for p in sorted((D / "tracked").glob("masks_shard*.pkl")):
        d = pickle.loads(p.read_bytes())
        out.update(d["masks"])
    return out


def cmd_weights(args) -> None:
    t = Test()
    res = {}
    for name, W in (("registered 0.40/0.20/0.20/0.20", W_REG), ("equal 0.25 each", W_FLAT)):
        runs = [t.run(s, W, 833) for s in SEEDS]
        res[name] = {k: float(np.mean([r[k] for r in runs])) for k in ("mIoU", "IoU", "mcIoU", "frames_all_right", "instance_accuracy")}
        res[name]["single_pass"] = {k: float(np.mean([r["single"][k] for r in runs])) for k in ("mIoU", "IoU", "mcIoU", "frames_all_right", "instance_accuracy")}
        res[name]["per_seed"] = [{k: r[k] for k in ("mIoU", "IoU", "mcIoU")} for r in runs]
        print(name, {k: round(100 * v, 2) for k, v in res[name].items() if isinstance(v, float)}, "single", {k: round(100 * v, 2) for k, v in res[name]["single_pass"].items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


def cmd_window(args) -> None:
    t = Test()
    off = load_offsets()
    ok = bad = 0

    def make_rows(k: int):
        def rows(i: int, fr: np.ndarray):
            nonlocal ok, bad
            m = off.get(i)
            if m is None:
                return None
            offs = [o for o in sorted(m) if o == 0 or mask_codec.decode(m[o][1]).any()]
            if len(offs) != fr.shape[0]:
                bad += 1
                return None
            ok += 1
            keep = np.array([abs(o) <= k for o in offs])
            return keep if keep.any() else None
        return rows

    res = {}
    for k in (0, 1, 2, 3, 5, 10):
        ok = bad = 0
        runs = [t.run(s, W_REG, 833, rows=make_rows(k)) for s in SEEDS]
        res[str(k)] = {m: float(np.mean([r[m] for r in runs])) for m in ("mIoU", "IoU", "mcIoU", "instance_accuracy")} | {"row_mapping_ok": ok, "row_mapping_bad": bad}
        print(f"+-{k} frames: mIoU {100 * res[str(k)]['mIoU']:.2f} IoU {100 * res[str(k)]['IoU']:.2f} mcIoU {100 * res[str(k)]['mcIoU']:.2f} acc {res[str(k)]['instance_accuracy']:.4f}  (mapped {ok}, mismatched {bad})", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


def cmd_unanimous(args) -> None:
    ex = REPO_ROOT / "experiments_edl" / "extract"
    res = {}
    for seed in SEEDS:
        c, e = load(ex / f"C_grasp_fold1_s{seed}.npz"), load(ex / f"E_grasp_fold1_s{seed}.npz")
        y = c["y"]
        top = lambda s: np.argsort(-s, kind="stable")[: int(round(0.2 * len(s)))]
        # softmax ensemble (dropout off): members' argmax
        logits = [c["det_" + m] for m in LABELS]
        arg = np.stack([z.argmax(1) for z in logits])
        p_soft = sum(w * softmax(z, axis=1) for w, z in zip([0.4, 0.2, 0.2, 0.2], logits))
        pred_soft = p_soft.argmax(1)
        U = (arg == arg[0]).all(0) & (pred_soft != y)
        ent = -(p_soft * np.log(p_soft + 1e-12)).sum(1)
        out = {"softmax max-softmax": (U, 1 - p_soft.max(1)), "softmax entropy": (U, ent)}
        # evidential
        alpha = sum(w * alpha_from_logits(e["det_" + m]) for w, m in zip([0.4, 0.2, 0.2, 0.2], LABELS))
        mu = alpha / alpha.sum(1, keepdims=True)
        arg_e = np.stack([e["det_" + m].argmax(1) for m in LABELS])
        Ue = (arg_e == arg_e[0]).all(0) & (mu.argmax(1) != y)
        out["evidential S1"] = (Ue, variance_scores(alpha)["epistemic"])
        out["evidential entropy"] = (Ue, -(mu * np.log(mu + 1e-12)).sum(1))
        # MC dropout vote
        cc = arm_c(c)
        Uc = cc["unanimous"]
        out["MC-dropout vote"] = (Uc, cc["signals"]["B1_mc_vote_disagreement"])
        for name, (U_, score) in out.items():
            res.setdefault(name, []).append({"unanimous_errors": int(U_.sum()), "found": float(U_[top(score)].sum() / max(1, U_.sum()))})
    summary = {n: {"unanimous_errors_mean": float(np.mean([r["unanimous_errors"] for r in v])), "found_at_20pct": {"mean": float(np.mean([r["found"] for r in v])), "values": [r["found"] for r in v]}} for n, v in res.items()}
    for n, v in summary.items():
        print(f"{n:22s} unanimous errors {v['unanimous_errors_mean']:.0f}  found in the top 20%: {100 * v['found_at_20pct']['mean']:.1f}%  {[round(100 * x, 1) for x in v['found_at_20pct']['values']]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["weights", "window", "unanimous"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"weights": cmd_weights, "window": cmd_window, "unanimous": cmd_unanimous}[a.cmd](a)
