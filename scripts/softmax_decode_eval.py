"""Single-pass softmax ensemble (cross-entropy members) with the argmax and the frame-decode readouts (GraSP official test, ground-truth masks, three seeds), for the appendix comparison with the
evidential ensemble. The evidential ensemble is read out the same way by scripts/decode_variants_eval.py ("none"). Accuracy and macro-F1 only.

    python scripts/softmax_decode_eval.py --dir experiments/gtbox_sam/gtmask --shift -0.5 --out softmax_decode.json
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
from scipy.special import softmax
from sklearn.metrics import f1_score

from evidential_analyze import LABELS, WEIGHTS
from frame_decode_eval import decode
from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--shift", type=float, default=-0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    train_labels = np.array([i[3] for i in GraspRegionDataset(args.data_root, "train", letterbox=True).instances])
    prior = np.bincount(train_labels, minlength=7) / len(train_labels)
    shift = -args.shift * np.log(prior)
    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    y = np.array([int(i[3]) for i in ds.instances])
    out: dict = {}
    for seed in args.seeds:
        ce = np.load(args.dir / f"ce_test_s{seed}.npz")
        pos = {int(i): k for k, i in enumerate(ce["index"])}
        assert len(pos) == len(y) and all(i in pos for i in range(len(y))), "the cross-entropy logits must cover every test instrument"
        take = np.array([pos[i] for i in range(len(y))])
        p = sum(w * softmax(ce["det_" + m], axis=1) for w, m in zip(WEIGHTS, LABELS))[take]
        rec = {}
        for name, pred in (("argmax", p.argmax(1)), ("decode", decode(p, by_frame, shift))):
            rec[name] = {"acc": 100 * float((pred == y).mean()), "F1": 100 * float(f1_score(y, pred, average="macro", labels=range(7)))}
        out[str(seed)] = rec
        print(f"seed {seed}: argmax acc {rec['argmax']['acc']:.2f} F1 {rec['argmax']['F1']:.2f} | decode acc {rec['decode']['acc']:.2f} F1 {rec['decode']['F1']:.2f}", flush=True)
    mean = {mode: {k: float(np.mean([out[s][mode][k] for s in out])) for k in ("acc", "F1")} for mode in ("argmax", "decode")}
    print("mean", json.dumps(mean))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"shift": args.shift, "per_seed": out, "mean": mean}, indent=1))


if __name__ == "__main__":
    main()
