"""Qualitative figure: one correction and one introduced error of the budget-matched gate, with the tracked crops and per-frame predictions.

For seed 42 and the instruments the budget-matched gate refines (top 941 by S_1, the `sel_s1_top833` column of the top-941 run of scripts/refinement_transitions.py),
the correction and the introduced error with the median S_1 are shown (a fixed rule, not a hand pick). The crops are made by re-running the SAM2.1-large tracker from
the stored keyframe mask; the per-frame labels and confidences are computed from the stored per-frame member logits of the paper's run (tracked_all_s42), so they
are the numbers behind the paper's fused label. The figure shows the keyframe and the neighbours nearest to the offsets -6, -3, +3 and +6 seconds that have a valid
track.

    python scripts/make_qualitative_figure.py --csv <refinement_transitions.csv> --final-dir experiments/gtbox_sam/final --out-dir <dir> \
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import build_track_frame_nums, crop_from_box, crop_from_mask
from evidential_seeds_e2e_eval import CONFIGS, ORDER
from gtbox_sam_final_eval import alpha_from_logits, fuse
from surgical_ai.data.region_dataset import GraspRegionDataset

OFFSETS = (-6, -3, 3, 6)


def pick(rows: list[dict], transition: str) -> dict:
    sel = sorted((r for r in rows if r["seed"] == "42" and r["sel_s1_top833"] == "True" and r["transition"] == transition), key=lambda r: float(r["s1"]))
    return sel[len(sel) // 2]


def load_det(final_dir: Path, idx: int):
    for f in sorted((final_dir / "tracked_all_s42").glob("frames_part*.npz")):
        z = np.load(f)
        if f"det_{idx}" in z.files:
            return z[f"det_{idx}"], int(z[f"center_{idx}"]), z[f"frames_{idx}"]
    raise KeyError(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--final-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sam2-checkpoint", type=Path, required=True)
    ap.add_argument("--sam2-config", required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from sam2.build_sam import build_sam2_video_predictor

    rows = list(csv.DictReader(open(args.csv)))
    chosen = {"correction": pick(rows, "wrong_to_correct"), "introduced error": pick(rows, "correct_to_wrong")}
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    names = ds.class_names_ordered()
    weights = CONFIGS["four"]
    init_masks = pickle.loads((args.final_dir / "masks.pkl").read_bytes())
    frames_root = args.data_root / "frames-001" / "frames"
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=args.device)
    tmp = args.out_dir / "tmp_track"
    report = {}
    panels = {}
    for kind, r in chosen.items():
        idx = int(r["instrument"])
        file_name, _seg, box, label = ds.instances[idx]
        case, stem = file_name.split("/")
        center_num = int(stem.replace(".jpg", ""))
        gt_mask = mask_codec.decode(init_masks[idx]).astype(bool)
        frame_nums, center_idx = build_track_frame_nums(frames_root, case, center_num, args.window)
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        for li, fn in enumerate(frame_nums):
            shutil.copy(frames_root / case / f"{fn:05d}.jpg", tmp / f"{li:05d}.jpg")
        state = predictor.init_state(video_path=str(tmp))
        predictor.add_new_mask(state, frame_idx=center_idx, obj_id=1, mask=gt_mask)
        masks = {center_idx: gt_mask}
        for fi, _o, ml in predictor.propagate_in_video(state):
            if fi != center_idx:
                masks[fi] = (ml[0, 0] > 0).cpu().numpy()
        for fi, _o, ml in predictor.propagate_in_video(state, reverse=True):
            if fi != center_idx:
                masks[fi] = (ml[0, 0] > 0).cpu().numpy()

        det, center_pos, local_idxs = load_det(args.final_dir, idx)
        alpha = sum(w * alpha_from_logits(det[:, ORDER.index(k), :]) for k, w in weights.items())
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        fused_label, fused_conf = fuse(det, weights)
        offs = {int(li) - center_idx: pos for pos, li in enumerate(local_idxs)}
        shown = [0]
        for target in OFFSETS:
            cand = [o for o in offs if o != 0 and np.sign(o) == np.sign(target)]
            if cand:
                shown.append(min(cand, key=lambda o: abs(o - target)))
        cols = []
        for o in shown:
            li = center_idx + o
            frame = np.array(Image.open(tmp / f"{li:05d}.jpg").convert("RGB"))
            crop = crop_from_box(frame, masks[li], box, True) if o == 0 else crop_from_mask(frame, masks[li], True)
            m = mu[offs[o]]
            cols.append({"offset": o, "crop": crop, "pred": names[int(m.argmax())], "conf": float(m.max())})
        panels[kind] = {"cols": cols, "gt": names[label], "single": names[int(mu[center_pos].argmax())], "single_conf": float(mu[center_pos].max()),
                        "fused": names[fused_label], "fused_conf": fused_conf, "case": case, "frame": stem, "instrument": idx, "n_frames": len(local_idxs)}
        report[kind] = {k: v for k, v in panels[kind].items() if k != "cols"} | {"shown_offsets": shown,
                                                                                  "per_frame": [(int(li) - center_idx, names[int(m.argmax())], round(float(m.max()), 3)) for li, m in zip(local_idxs, mu)]}
    shutil.rmtree(tmp, ignore_errors=True)
    (args.out_dir / "qualitative_report.json").write_text(json.dumps(report, indent=1))

    import textwrap

    from matplotlib.gridspec import GridSpec

    def trim(img: np.ndarray, margin: int = 3) -> np.ndarray:
        """Drop the black letterbox padding so the instrument fills its tile."""
        ys, xs = np.nonzero(img.sum(axis=2) > 24)
        if len(ys) == 0:
            return img
        return img[max(0, ys.min() - margin): ys.max() + 1 + margin, max(0, xs.min() - margin): xs.max() + 1 + margin]

    def square(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        side = max(h, w)
        out = np.zeros((side, side, 3), img.dtype)
        out[(side - h) // 2: (side - h) // 2 + h, (side - w) // 2: (side - w) // 2 + w] = img
        return out

    ncol = max(len(p["cols"]) for p in panels.values())
    tiles = {k: [square(trim(c["crop"])) for c in p["cols"]] for k, p in panels.items()}
    ratio = [1.0] * ncol
    fig = plt.figure(figsize=(6.6, 2.55))
    gs = GridSpec(2, ncol + 1, figure=fig, width_ratios=[1.3] + ratio, left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.04, hspace=0.40)
    wrap = lambda t, w=17: "\n".join(textwrap.wrap(t, w))
    for row, (kind, p) in enumerate(panels.items()):
        lab = fig.add_subplot(gs[row, 0])
        lab.axis("off")
        lab.text(0.0, 0.5, f"{kind.capitalize()}\n" + wrap(f"single pass: {p['single']} ({p['single_conf']:.2f})") + "\n" + wrap(f"fused: {p['fused']} ({p['fused_conf']:.2f})") + "\n" + wrap(f"true: {p['gt']}"),
                 ha="left", va="center", fontsize=5.6, linespacing=1.3)
        for c in range(ncol):
            ax = fig.add_subplot(gs[row, c + 1])
            ax.axis("off")
            if c >= len(p["cols"]):
                continue
            col = p["cols"][c]
            ax.imshow(tiles[kind][c])
            title = ("keyframe" if col["offset"] == 0 else f"{col['offset']:+d} s") + "\n" + wrap(col["pred"], 26) + f" {col['conf']:.2f}"
            ax.set_title(title, fontsize=5.4, color="#1f2933" if col["pred"] == p["gt"] else "#c0392b", pad=2, linespacing=1.2)
    fig.savefig(args.out_dir / "fig_qualitative.pdf", dpi=300)
    fig.savefig(args.out_dir / "fig_qualitative.png", dpi=200)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "per_frame"} for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
