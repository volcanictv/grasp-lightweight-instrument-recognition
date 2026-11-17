"""mIoU / IoU / mcIoU, instance accuracy and macro-F1 of the evidential ensemble's single pass and of each single member, for
logits extracted from one mask source (SAM masks or the ground-truth masks) and painted onto that same mask source. Used for the
mask-quality upper bound of the recognition ablation: run once with the SAM masks (must reproduce the paper's numbers), once with
ground-truth masks.

Usage: python scripts/member_scores_on_masks.py --masks M.pkl --logits logits_s42.npz [logits_s43.npz ...] --out result.json
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
from sklearn.metrics import f1_score

from evidential_seeds_e2e_eval import CONFIGS, ORDER, alpha_mix
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.evaluation.semantic_iou import aggregate, frame_class_ious, paint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--logits", type=Path, nargs="+", required=True, help="one npz per seed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    masks = pickle.loads(args.masks.read_bytes())
    zs = [np.load(p) for p in args.logits]
    y = zs[0]["y"]

    def softmax(l: np.ndarray) -> np.ndarray:
        e = np.exp(l - l.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    # the score only orders overlapping masks when painting: belief for the ensemble, softmax confidence for a member
    mus = [(lambda a: a / a.sum(1, keepdims=True))(alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"])) for z in zs]
    configs: dict[str, np.ndarray] = {"oracle classes": np.tile(y, (len(zs), 1))}
    scores: dict[str, np.ndarray] = {"oracle classes": np.ones((len(zs), len(y)))}
    configs["ensemble single pass"] = np.stack([m.argmax(1) for m in mus])
    scores["ensemble single pass"] = np.stack([m.max(1) for m in mus])
    for member in ORDER:
        configs[f"member {member}"] = np.stack([z[f"det_{member}"].argmax(1) for z in zs])
        scores[f"member {member}"] = np.stack([softmax(z[f"det_{member}"]).max(1) for z in zs])
    done = np.all([z["done"] for z in zs], axis=0)

    by_frame: dict[str, list[int]] = defaultdict(list)
    for idx, (file_name, _s, _b, _l) in enumerate(ds.instances):
        by_frame[file_name].append(idx)
    per = {(name, s): [] for name in configs for s in range(len(zs))}
    for n, (file_name, indices) in enumerate(by_frame.items()):
        shape = tuple(ds.instances[indices[0]][1]["size"])
        gt = paint(shape, [(decode_instance_mask(ds.instances[i][1]).astype(bool), int(y[i]) + 1, 0.0) for i in indices])
        used = {i: mask_codec.decode(masks[i]).astype(bool) for i in indices if i in masks and done[i]}
        for name, preds in configs.items():
            for s in range(len(zs)):
                insts = [(used[i], int(preds[s][i]) + 1, float(scores[name][s][i])) for i in indices if i in used]
                per[(name, s)].append(frame_class_ious(paint(shape, insts), gt))
        if (n + 1) % 200 == 0:
            print(f"{n + 1}/{len(by_frame)} frames", flush=True)

    out: dict = {"instruments": int(len(y)), "classified": int(done.sum()), "seeds": [str(p) for p in args.logits], "configs": {}}
    for name, preds in configs.items():
        runs = []
        for s in range(len(zs)):
            r = aggregate(per[(name, s)])
            r["instance_accuracy"] = float((preds[s][done] == y[done]).mean())
            r["instance_macro_f1"] = float(f1_score(y[done], preds[s][done], average="macro", labels=range(7)))
            runs.append(r)
        mean = {k: float(np.mean([r[k] for r in runs])) for k in ("mIoU", "IoU", "mcIoU", "instance_accuracy", "instance_macro_f1")}
        out["configs"][name] = {"mean": mean, "per_seed": runs}
        print(f"{name:<28}" + "".join(f"{100 * mean[k]:>9.2f}" for k in mean))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
