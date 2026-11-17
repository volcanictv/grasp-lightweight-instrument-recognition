"""Qualitative figure of the TAPIS comparison: one instrument per challenge group, with the keyframe, the ground truth, TAPIS, our single pass, our final label and the tracked frames.

The instruments are picked by a fixed rule, not by hand (seed 42, the budget-matched top-941 gate):
  ours better   TAPIS wrong, our final label right in all three seeds, our single pass wrong in seed 42 and the instrument sent to tracking (so the temporal frames are
                what changed the label); within the group's union flag, the candidate with the median S1 of those (an instrument already shown in an earlier row is skipped)
  TAPIS better TAPIS right, our final label wrong in all three seeds; within the stated flag, the median-S1 candidate
Groups: occlusion_any, degraded_any, boundary_any, GT class in a confusable pair (ours better); specular and the Bipolar/Prograsp pair (TAPIS better).

    python scripts/challenge_strata_figure.py picks --per-instrument <csv> --transitions <csv> --out picks.json
    python scripts/challenge_strata_figure.py fig --picks picks.json --final-dir experiments/gtbox_sam/final --out-dir <dir> --sam2-checkpoint ... --sam2-config ...   (titanxp, GPU)
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

import numpy as np

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
SHORT = ["Bipolar", "Prograsp", "Needle driver", "Scissors", "Suction", "Clip applier", "Grasper"]
OFFSETS = (-6, -3, 3, 6)
ROWS = [("better", "Occlusion / missing parts", "occlusion_any"), ("better", "Blood, glare, haze", "degraded_any"), ("better", "Frame boundary", "boundary_any"),
        ("better", "Similar classes", "pair_class"), ("worse", "Glare (TAPIS better)", "specular"), ("worse", "Bipolar / Prograsp (TAPIS better)", "bp_pair")]


def picks(args) -> None:
    P = list(csv.DictReader(open(args.per_instrument)))
    T = {int(r["instrument"]): r for r in csv.DictReader(open(args.transitions)) if r["seed"] == "42"}
    n = len(P)
    a = lambda k: np.array([int(r[k]) for r in P])
    gt, tap = a("gt"), a("tapis")
    fin = [a(f"final_s{s}") for s in (42, 43, 44)]
    sin42 = a("single_s42")
    sel42 = np.array([T[i]["sel_s1_top833"] == "True" for i in range(n)])
    s1 = np.array([float(T[i]["s1"]) for i in range(n)])
    all_fin_ok = (fin[0] == gt) & (fin[1] == gt) & (fin[2] == gt)
    all_fin_bad = (fin[0] != gt) & (fin[1] != gt) & (fin[2] != gt)
    better = (tap != gt) & all_fin_ok & (sin42 != gt) & sel42
    worse = (tap == gt) & all_fin_bad
    flag = {"occlusion_any": a("occlusion_any") > 0, "degraded_any": a("degraded_any") > 0, "boundary_any": a("boundary_any") > 0, "pair_class": np.isin(gt, [0, 1, 4, 6]),
            "specular": a("specular") > 0, "bp_pair": np.isin(gt, [0, 1])}
    out, used = [], set()
    for kind, title, fl in ROWS:
        cand = np.array([c for c in np.where((better if kind == "better" else worse) & flag[fl])[0] if int(c) not in used])
        cand = cand[np.argsort(s1[cand], kind="stable")]
        i = int(cand[len(cand) // 2])
        used.add(i)
        out.append({"kind": kind, "title": title, "flag": fl, "instrument": i, "n_candidates": int(len(cand)), "file": P[i]["file"], "gt": int(gt[i]), "tapis": int(tap[i]),
                    "single": int(sin42[i]), "final": int(fin[0][i]), "s1": float(s1[i]), "box": [int(P[i][k]) for k in ("box_x", "box_y", "box_w", "box_h")]})
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def fig(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from PIL import Image
    from pycocotools import mask as mask_codec

    from evaluate_temporal_track_ensemble import build_track_frame_nums, crop_from_box, crop_from_mask
    from evidential_seeds_e2e_eval import CONFIGS, ORDER
    from gtbox_sam_final_eval import alpha_from_logits, fuse
    from make_qualitative_figure import load_det
    from sam2.build_sam import build_sam2_video_predictor
    from surgical_ai.data.region_dataset import GraspRegionDataset

    data_root = args.data_root
    chosen = json.loads(Path(args.picks).read_text())
    ds = GraspRegionDataset(data_root, "test", letterbox=True)
    weights = CONFIGS["four"]
    init_masks = pickle.loads((args.final_dir / "masks.pkl").read_bytes())
    frames_root = data_root / "frames-001" / "frames"
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=args.device)
    tmp = args.out_dir / "tmp_track"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panels, report = [], []
    for pk in chosen:
        idx = pk["instrument"]
        file_name, seg, box, label = ds.instances[idx]
        assert label == pk["gt"] and list(box) == pk["box"], "pick does not match the dataset instrument"
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
        for rev in (False, True):
            for fi, _o, ml in predictor.propagate_in_video(state, reverse=rev):
                if fi != center_idx:
                    masks[fi] = (ml[0, 0] > 0).cpu().numpy()
        det, center_pos, local_idxs = load_det(args.final_dir, idx)
        alpha = sum(w * alpha_from_logits(det[:, ORDER.index(k), :]) for k, w in weights.items())
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        fused_label, fused_conf = fuse(det, weights)
        offs = {int(li) - center_idx: pos for pos, li in enumerate(local_idxs)}
        shown = []
        for target in OFFSETS:
            cand = [o for o in offs if o != 0 and np.sign(o) == np.sign(target)]
            if cand:
                shown.append(min(cand, key=lambda o: abs(o - target)))
        tiles = []
        for o in shown:
            li = center_idx + o
            fr = np.array(Image.open(tmp / f"{li:05d}.jpg").convert("RGB"))
            crop = crop_from_mask(fr, masks[li], True)
            m = mu[offs[o]]
            tiles.append({"offset": o, "crop": crop, "pred": int(m.argmax()), "conf": float(m.max())})
        key_frame = np.array(Image.open(frames_root / case / f"{center_num:05d}.jpg").convert("RGB"))
        key_crop = crop_from_box(key_frame, gt_mask, box, True)
        per_frame = [int(m.argmax()) for m in mu]
        panels.append({**pk, "tiles": tiles, "key_frame": key_frame, "key_crop": key_crop, "fused": fused_label, "fused_conf": fused_conf, "n_frames": len(per_frame),
                       "n_frames_correct": int(sum(p == label for p in per_frame)), "gt_mask": gt_mask})
        report.append({k: v for k, v in panels[-1].items() if k not in ("tiles", "key_frame", "key_crop", "gt_mask")}
                      | {"per_frame": [(int(li) - center_idx, SHORT[int(m.argmax())], round(float(m.max()), 3)) for li, m in zip(local_idxs, mu)]})
    shutil.rmtree(tmp, ignore_errors=True)
    (args.out_dir / "figure_report.json").write_text(json.dumps(report, indent=1))

    def trim(img: np.ndarray, margin: int = 3) -> np.ndarray:
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

    ok, bad = "#1b7f3b", "#c0392b"
    ncol = 4
    fig = plt.figure(figsize=(10.4, 1.78 * len(panels) + 0.5))
    gs = GridSpec(len(panels), 3 + ncol, figure=fig, width_ratios=[2.1, 1.05, 1.75] + [1.0] * ncol, left=0.005, right=0.995, top=0.945, bottom=0.005, wspace=0.05, hspace=0.30)
    heads = ["Keyframe (GT box)", "Classifier input (GT)", "Predictions"] + [f"Tracked frame {o:+d} s" for o in (-6, -3, 3, 6)]
    for r, p in enumerate(panels):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(p["key_frame"])
        x, y, w, h = p["box"]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="#f1c40f", lw=1.6))
        ax.axis("off")
        ax.text(0.01, 0.99, f"{p['title']}\n{p['file']}", transform=ax.transAxes, va="top", ha="left", fontsize=6.2, color="white", bbox=dict(fc="black", alpha=0.6, pad=1.5, lw=0))
        ax = fig.add_subplot(gs[r, 1])
        ax.imshow(square(trim(p["key_crop"])))
        ax.axis("off")
        ax.set_title(f"GT: {SHORT[p['gt']]}", fontsize=6.5, pad=2)
        ax = fig.add_subplot(gs[r, 2])
        ax.axis("off")
        lines = [("TAPIS (GT box)", p["tapis"]), ("Ours, single pass", p["single"]), ("Ours, final", p["final"])]
        for k, (nm, c) in enumerate(lines):
            ax.text(0.0, 0.84 - 0.27 * k, f"{nm}:\n{SHORT[c]}", fontsize=6.5, va="center", color=ok if c == p["gt"] else bad, linespacing=1.2)
        ax.text(0.0, 0.03, f"{p['n_frames_correct']}/{p['n_frames']} tracked frames vote GT", fontsize=5.6, va="bottom", color="#444")
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, 3 + c])
            ax.axis("off")
            if c < len(p["tiles"]):
                t = p["tiles"][c]
                ax.imshow(square(trim(t["crop"])))
                ax.set_title(f"{t['offset']:+d} s: {SHORT[t['pred']]} {t['conf']:.2f}", fontsize=5.8, pad=2, color=ok if t["pred"] == p["gt"] else bad)
    for c, h in enumerate(heads):
        fig.text([0.105, 0.31, 0.43, 0.575, 0.69, 0.805, 0.92][c], 0.968, h, ha="center", fontsize=7, weight="bold")
    fig.savefig(args.out_dir / "fig_challenge_strata.pdf", dpi=300)
    fig.savefig(args.out_dir / "fig_challenge_strata.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("picks")
    p.add_argument("--per-instrument", type=Path, required=True)
    p.add_argument("--transitions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fig")
    f.add_argument("--picks", type=Path, required=True)
    f.add_argument("--final-dir", type=Path, required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--sam2-checkpoint", type=Path, required=True)
    f.add_argument("--sam2-config", required=True)
    f.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    f.add_argument("--window", type=int, default=10)
    f.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    picks(args) if args.cmd == "picks" else fig(args)


if __name__ == "__main__":
    main()
