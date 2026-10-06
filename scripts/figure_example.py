"""Real-frame data for the paper's figures: for one test instrument, the frames of its track (the masks of the final run), the per-frame class beliefs of the headline
seed's four-member evidential ensemble, and the keyframe's single-pass belief and epistemic score. Used by scripts/make_paper_figures.py; run directly it writes a contact
sheet of candidates so the example for Figure 1 can be chosen by eye.

Usage (titanxp, surgical environment, repo root; CPU only):
    python scripts/figure_example.py --sheet experiments/figure_examples/sheets --limit 12
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import crop_from_box, crop_from_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.models import build_model

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}
RUN = REPO_ROOT / "experiments" / "gtbox_sam" / "final"
DATA_ROOT = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))


def load_members(config: Path, device: str = "cpu"):
    cfg = yaml.safe_load(config.read_text())
    members = []
    for m in cfg["members"]:
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        members.append((m["label"], net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))
    return members


def belief(member_logits: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    alpha = sum(WEIGHTS[k] * alpha_from_logits(v[None])[0] for k, v in member_logits.items())
    mu = alpha / alpha.sum()
    return mu, float(variance_scores(alpha[None])["epistemic"][0])


def load_tracks() -> dict:
    masks = {}
    for p in sorted((RUN / "tracked").glob("masks_shard*.pkl")):
        masks.update(pickle.loads(p.read_bytes())["masks"])
    return masks


def example(index: int, members, ds: GraspRegionDataset, tracks: dict, max_frames: int | None = None) -> dict:
    file_name, seg, box, label = ds.instances[index]
    case = file_name.split("/")[0]
    frames_root = DATA_ROOT / "frames-001" / "frames"
    out = []
    for offset, (frame_num, rle) in sorted(tracks[index].items()):
        image = np.array(Image.open(frames_root / case / f"{frame_num:05d}.jpg").convert("RGB"))
        mask = mask_codec.decode(rle).astype(bool)
        if not mask.any():
            continue
        ml = {}
        ok = True
        for name, net, tf, letterbox in members:
            crop = crop_from_box(image, mask, box, letterbox) if offset == 0 else crop_from_mask(image, mask, letterbox)
            if crop is None:
                ok = False
                break
            with torch.no_grad():
                ml[name] = net(tf(Image.fromarray(crop)).unsqueeze(0)).float().numpy()[0]
        if not ok:
            continue
        mu, s1 = belief(ml)
        out.append({"offset": int(offset), "frame": int(frame_num), "image": image, "mask": mask, "mu": mu, "s1": s1})
    # evidence-weighted fusion of the frames, as in the pipeline
    mus = np.stack([f["mu"] for f in out])
    combined = (mus.max(axis=1)[:, None] * mus).sum(axis=0)
    key = next(f for f in out if f["offset"] == 0)
    return {"index": index, "file": file_name, "box": box, "true": CLASSES[label], "true_index": label, "frames": out, "single": key["mu"], "s1": key["s1"],
            "fused": combined / combined.sum()}


def sheet(ex: dict, path: Path, n_panels: int = 7) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = ex["frames"]
    pick = np.unique(np.linspace(0, len(frames) - 1, n_panels).round().astype(int))
    center = [i for i, f in enumerate(frames) if f["offset"] == 0][0]
    pick = sorted(set(pick.tolist()) | {center})
    x, y, w, h = ex["box"]
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * 1.7
    fig, axes = plt.subplots(1, len(pick), figsize=(2.3 * len(pick), 2.9))
    for ax, i in zip(np.atleast_1d(axes), pick):
        f = frames[i]
        H, W = f["image"].shape[:2]
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        ax.imshow(f["image"][y0:y1, x0:x1])
        ax.contour(f["mask"][y0:y1, x0:x1].astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.6)
        ax.axis("off")
        k = int(np.argmax(f["mu"]))
        ax.set_title(f"t={f['offset']:+d}\n{CLASSES[k][:14]} {f['mu'][k]:.2f}", fontsize=7, color="green" if k == ex["true_index"] else "red")
    fig.suptitle(f"#{ex['index']} {ex['file']} true {ex['true']} | keyframe S1 {ex['s1']:.1e} | fused {CLASSES[int(np.argmax(ex['fused']))]} {ex['fused'].max():.2f}", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=REPO_ROOT / "experiments" / "figure_examples" / "candidates.json")
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--indices", type=int, nargs="*", default=None, help="render these instrument indices instead of the filtered list")
    ap.add_argument("--min-side", type=int, default=70)
    ap.add_argument("--min-share", type=float, default=0.5)
    args = ap.parse_args()
    torch.set_num_threads(2)
    cands = json.loads(args.candidates.read_text())
    keep = []
    for c in cands:
        x, y, w, h = c["box"]
        if x > 4 and y > 4 and x + w < 1274 and y + h < 796 and min(w, h) >= args.min_side * 0.6 and max(w, h) >= args.min_side and c["track_frames"] >= 15 and c["fused_share"] >= args.min_share:
            keep.append(c)
    print(len(keep), "candidates after the filters")
    members = load_members(REPO_ROOT / "configs" / "arms" / "ens4_N_official_s44.yaml")
    ds = GraspRegionDataset(DATA_ROOT, "test", letterbox=True)
    tracks = load_tracks()
    if args.indices:
        keep = [c for c in cands if c["index"] in args.indices]
    for c in keep[: args.limit]:
        ex = example(c["index"], members, ds, tracks)
        sheet(ex, args.sheet / f"cand_{c['index']}.png")
        print("wrote", c["index"], c["file"], c["true"], "<-", c["single_frame"], flush=True)


if __name__ == "__main__":
    main()
