"""End-to-end check of the vote-based classifier on predicted masks, scored with
the segmentation metric the GraSP paper reports (AP50_segm, mAP@0.5IoU_segm).

Detections come from the official-split Mask R-CNN (score floor 0.05, the same as
evaluate_classifier_swap_ap50.py). Each detection's masked box crop is classified
by the deep-dropout vote ensemble: 20 dropout-on passes per member, weighted
plurality of the class votes, u = 1 - top weighted vote share. No softmax is used
for the label or for any score. No SAM2 tracking is included in this run.

Variants (fixed before any number was seen, see docs/reports/end_to_end/summary.md):
  a  detector label, detector score (must reproduce the recorded 0.8101)
  b  vote label, detector score unchanged (the earlier failure mode)
  c  vote label, score = detector score x top weighted vote share
  d  as c, and detections with vote uncertainty u > 0.5 dropped

Usage:
    python scripts/evaluate_end_to_end_vote_ap50.py \\
        --maskrcnn-checkpoint experiments/instance_segmentation_maskrcnn_official_20260901-180758/best.pt \\
        --data-root GraSP --device cuda:0 --out docs/reports/end_to_end/results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image

from analyze_uncertainty_signals import enable_mc_dropout
from build_tracking_sample_b import plurality
from surgical_ai.data.detection_dataset import GraspDetectionDataset, build_detection_transforms
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.segmentation import evaluate_instance_ap50
from surgical_ai.models import build_model
from surgical_ai.models.detectors.registry import build_detector

MC = 20
CHUNK = {320: 6, 224: 12}


def pad_to_square(crop: np.ndarray) -> np.ndarray:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    top, left = (side - ch) // 2, (side - cw) // 2
    square[top:top + ch, left:left + cw] = crop
    return square


def member_votes(model, tf, crops: list[np.ndarray], size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(mc vote share (D, C), dropout-off one-hot vote (D, C)) for one member."""
    mc_out, det_out = [], []
    step = CHUNK[size]
    for i in range(0, len(crops), step):
        x = torch.stack([tf(Image.fromarray(c)) for c in crops[i:i + step]]).to(device)
        model.eval()
        det = model(x).argmax(1).cpu().numpy()
        enable_mc_dropout(model)
        logits = model(x.repeat_interleave(MC, dim=0)).view(len(x), MC, -1)
        model.eval()
        votes = logits.argmax(2).cpu().numpy()
        mc_out.append((votes[..., None] == np.arange(7)).mean(axis=1))
        det_out.append((det[:, None] == np.arange(7)).astype(float))
    return np.concatenate(mc_out), np.concatenate(det_out)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maskrcnn-checkpoint", type=Path, required=True)
    ap.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-threshold", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ds = GraspDetectionDataset(args.data_root, args.split, transform=build_detection_transforms(train=False))
    class_names = ds.class_names_ordered()
    maskrcnn = build_detector("maskrcnn_mobilenet_v3", num_classes=7, pretrained=False).to(device)
    maskrcnn.load_state_dict(torch.load(args.maskrcnn_checkpoint, map_location=device))
    maskrcnn.eval()

    cfg = yaml.safe_load(args.ensemble_config.read_text())
    w320 = cfg["weight_resnet50_320"]
    members = []
    for m in cfg["members"]:
        model = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()
        w = w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(cfg["members"]) - 1)
        members.append((m, model, build_transforms(m["image_size"], train=False), w))
    wsum = sum(w for *_x, w in members)

    to_tensor = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    n = len(ds.samples) if args.limit is None else min(args.limit, len(ds.samples))
    variants = {k: [] for k in "abcd"}
    gts_all = []
    n_det = 0
    for idx in range(n):
        file_name, anns = ds.samples[idx]
        frame = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        height, width = frame.shape[:2]
        out = maskrcnn([to_tensor(Image.fromarray(frame)).to(device)])[0]
        keep = [i for i, s in enumerate(out["scores"].cpu().numpy()) if s >= args.score_threshold]
        boxes = out["boxes"].cpu().numpy()[keep]
        labels = out["labels"].cpu().numpy()[keep] - 1
        scores = out["scores"].cpu().numpy()[keep]
        masks = out["masks"].cpu().numpy()[keep, 0] >= 0.5

        crops, valid = [], []
        for j, (box, mask) in enumerate(zip(boxes, masks)):
            x1, y1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
            x2, y2 = min(width, int(round(box[2]))), min(height, int(round(box[3])))
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append((frame[y1:y2, x1:x2] * mask[y1:y2, x1:x2, None]).astype(np.uint8))
            valid.append(j)
        v_mc = np.zeros((len(crops), 7))
        v_det = np.zeros((len(crops), 7))
        if crops:
            for m, model, tf, w in members:
                cs = [pad_to_square(c) for c in crops] if m["letterbox"] else crops
                mc, det = member_votes(model, tf, cs, m["image_size"], device)
                v_mc += (w / wsum) * mc
                v_det += (w / wsum) * det
        vote_label = dict(zip(valid, plurality(v_mc, v_det))) if crops else {}
        top = dict(zip(valid, v_mc.max(axis=1))) if crops else {}

        img = {k: [] for k in "abcd"}
        for j in range(len(scores)):
            mask, s = masks[j], float(scores[j])
            img["a"].append((mask, int(labels[j]), s))
            if j not in vote_label:
                for k in "bcd":
                    img[k].append((mask, int(labels[j]), s))
                continue
            lab, share = int(vote_label[j]), float(top[j])
            img["b"].append((mask, lab, s))
            img["c"].append((mask, lab, s * share))
            if 1.0 - share <= 0.5:
                img["d"].append((mask, lab, s * share))
        for k in "abcd":
            variants[k].append(img[k])
        n_det += len(scores)
        gts = []
        for a in anns:
            gm = decode_instance_mask(a["segmentation"]).astype(bool)
            if gm.any():
                gts.append((gm, ds._id_to_index[a["category_id"]]))
        gts_all.append(gts)
        if idx % 50 == 0:
            print(f"{idx}/{n} frames, {n_det} detections", flush=True)

    for _ in range(len(ds.samples) - n):
        for k in "abcd":
            variants[k].append([])
        gts_all.append([])

    result = {"n_frames": n, "n_detections": n_det, "score_threshold": args.score_threshold, "variants": {}}
    names = {"a": "detector label, detector score", "b": "vote label, detector score",
             "c": "vote label, score = detector score x top vote share",
             "d": "as c, detections with vote uncertainty > 0.5 dropped"}
    for k in "abcd":
        r = evaluate_instance_ap50(variants[k], gts_all, class_names)
        result["variants"][k] = {"description": names[k], "ap50_segm": float(r["map50"]),
                                 "per_class_ap50": {c: float(v) for c, v in r["per_class_ap50"].items()},
                                 "n_kept_detections": int(sum(len(x) for x in variants[k]))}
        print(f"variant {k} ({names[k]}): AP50_segm {r['map50']:.4f}", flush=True)
        for c, v in r["per_class_ap50"].items():
            print(f"   {c:<28} {v:.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
