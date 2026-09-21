"""Optical-flow tracking baseline in the spirit of ISINet's temporal consistency
module (Gonzalez et al., MICCAI 2020). This is a flow-propagation baseline, NOT
ISINet: ISINet's own trained classifier and flow network are not used.

Design, fixed before any run:
- Population: the official-test instances with vote disagreement u >= 0.09 under
  the shipped deep-dropout vote ensemble (configs/region_ensemble_deepdropout.yaml,
  experiments/mc_logits_cache_deepdropout.npz): the same 833 instances SAM2
  tracking is scored on (0.9623 accuracy / 0.9351 macro-F1, 131 fixed, 29 broken).
- Propagation: from the annotated mask at the centre frame, chain dense optical
  flow (torchvision RAFT-large, cached C_T_SKHT_V2 weights, computed at half
  resolution and upsampled) between consecutive frames up to 10 frames each way,
  over the same frame windows as SAM2 tracking (build_track_frame_nums). The mask
  is pulled back through the flow field frame by frame (bilinear, threshold 0.5).
  A frame is dropped, and that direction stops, when the warped mask is empty;
  the SAM2 pipeline likewise drops only empty masks.
- Classification: as evaluate_temporal_track_ensemble.py (crop_from_box for the
  centre, crop_from_mask elsewhere, letterbox per member, 20 MC passes per member
  per frame, weighted vote share).
- Combination, three rules on the same frames: plain majority over frame labels
  (the ISINet-like rule), pooled vote, uncertainty-weighted vote (our rule,
  evaluate_softmax_free_combine.track_predictions).
- Final prediction: the combined result for flagged instances, the ensemble's own
  vote for the rest. Scored against the ensemble alone and against SAM2 tracking
  with the same weighted vote (paired exact McNemar). One seed, single run.

Usage (titanxp, cuda:1 only):
    python scripts/evaluate_flow_tracking_baseline.py run --shard-id 0 --num-shards 2 --device cuda:1 --data-root GraSP
    python scripts/evaluate_flow_tracking_baseline.py score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from analyze_uncertainty_signals import enable_mc_dropout
from build_tracking_sample_b import plurality, vote_share
from evaluate_softmax_free_combine import RULES, track_predictions
from evaluate_temporal_track_ensemble import build_track_frame_nums, crop_from_box, crop_from_mask
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

GATE = 0.09
WINDOW = 10
MC = 20
OUT_DIR = REPO_ROOT / "experiments" / "flow_baseline"
CONFIG = REPO_ROOT / "configs" / "region_ensemble_deepdropout.yaml"
CACHE = REPO_ROOT / "experiments" / "mc_logits_cache_deepdropout.npz"
SAM2_DIRS = [REPO_ROOT / "docs" / "reports" / "tracking_softmax_free", REPO_ROOT / "docs" / "reports" / "tracking_gate_sweep"]


def ensemble_state():
    cfg = yaml.safe_load(CONFIG.read_text())
    members, w320 = cfg["members"], cfg["weight_resnet50_320"]
    weights = np.array([w320 if m["label"] == "resnet50_320" else (1 - w320) / (len(members) - 1) for m in members])
    weights = weights / weights.sum()
    cache = np.load(CACHE)
    y = cache["y_true"]
    n_classes = cache["det_" + members[0]["label"]].shape[1]
    v_mc = sum(w * vote_share(cache[f"mc_{m['label']}"], n_classes) for w, m in zip(weights, members))
    v_det = sum(w * vote_share(cache[f"det_{m['label']}"], n_classes) for w, m in zip(weights, members))
    base = plurality(v_mc, v_det)
    u = np.round(1 - v_mc.max(axis=1), 9)
    return members, weights, y, base, u, n_classes


class FlowCache:
    """RAFT flow between consecutive frames, half resolution, small LRU."""

    def __init__(self, device, frames_root: Path, capacity: int = 240):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).to(device).eval()
        self.device, self.root, self.capacity = device, frames_root, capacity
        self.images: OrderedDict = OrderedDict()
        self.flows: OrderedDict = OrderedDict()

    def image(self, case: str, num: int) -> np.ndarray:
        key = (case, num)
        if key not in self.images:
            self.images[key] = np.array(Image.open(self.root / case / f"{num:05d}.jpg").convert("RGB"))
            if len(self.images) > 64:
                self.images.popitem(last=False)
        self.images.move_to_end(key)
        return self.images[key]

    def _half(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        t = F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=False)
        return t * 2 - 1

    @torch.no_grad()
    def flow(self, case: str, src: int, dst: int) -> torch.Tensor:
        """Flow field that, for each pixel of frame `src`, points into frame `dst` (half res, (1,2,H,W))."""
        key = (case, src, dst)
        if key not in self.flows:
            a, b = self._half(self.image(case, src)), self._half(self.image(case, dst))
            self.flows[key] = self.model(a, b)[-1].half().cpu()
            if len(self.flows) > self.capacity:
                self.flows.popitem(last=False)
        self.flows.move_to_end(key)
        return self.flows[key].to(self.device).float()


def pull_mask(mask_prev: torch.Tensor, flow_cur_to_prev: torch.Tensor) -> torch.Tensor:
    """mask_prev: (1,1,H,W) float in full res; flow at half res. Returns the mask in the current frame."""
    h, w = mask_prev.shape[-2:]
    flow = F.interpolate(flow_cur_to_prev, size=(h, w), mode="bilinear", align_corners=False) * 2.0
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    gx = (xs[None].float() + flow[:, 0]) / (w - 1) * 2 - 1
    gy = (ys[None].float() + flow[:, 1]) / (h - 1) * 2 - 1
    return F.grid_sample(mask_prev, torch.stack([gx, gy], dim=-1), mode="bilinear", padding_mode="zeros", align_corners=True)


def propagate(fc: FlowCache, case: str, nums: list[int], center_idx: int, gt_mask: np.ndarray, device) -> dict[int, np.ndarray]:
    masks = {center_idx: gt_mask}
    for step in (1, -1):
        prev = torch.from_numpy(gt_mask.astype(np.float32))[None, None].to(device)
        i = center_idx + step
        while 0 <= i < len(nums):
            cur = pull_mask(prev, fc.flow(case, nums[i], nums[i - step]))
            if (cur > 0.5).sum() == 0:
                break
            prev = (cur > 0.5).float()
            masks[i] = prev[0, 0].cpu().numpy().astype(bool)
            i += step
    return masks


def run(args) -> None:
    members_cfg, weights, y, base, u, n_classes = ensemble_state()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    data_root = args.data_root
    frames_root = data_root / "frames-001" / "frames"
    ds = GraspRegionDataset(data_root, "test", letterbox=True)
    assert len(ds.instances) == len(y)
    flagged = np.where(u >= GATE)[0]
    mine = flagged[args.shard_id::args.num_shards]
    if args.limit:
        mine = mine[:args.limit]
    print(f"{len(flagged)} flagged; this shard {len(mine)}", flush=True)

    models = []
    for m in members_cfg:
        model = build_model(m["model"], num_classes=n_classes, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        model.eval()
        models.append((model, build_transforms(m["image_size"], train=False), m["letterbox"]))
    fc = FlowCache(device, frames_root)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_out: dict = {}
    meta = []
    for n, idx in enumerate(mine):
        t0 = time.time()
        file_name, segmentation, box, label = ds.instances[idx]
        case, stem = file_name.split("/")
        center_num = int(stem.replace(".jpg", ""))
        gt_mask = decode_instance_mask(segmentation).astype(bool)
        nums, center_idx = build_track_frame_nums(frames_root, case, center_num, WINDOW)
        t_flow = time.time()
        masks = propagate(fc, case, nums, center_idx, gt_mask, device)
        t_flow = time.time() - t_flow
        frame_arrays = {i: fc.image(case, nums[i]) for i in masks}
        crops: dict = {}

        def get_crop(i: int, lb: bool):
            if (i, lb) not in crops:
                crops[(i, lb)] = (crop_from_box(frame_arrays[i], masks[i], box, lb) if i == center_idx
                                  else crop_from_mask(frame_arrays[i], masks[i], lb))
            return crops[(i, lb)]

        valid = [i for i in sorted(masks) if get_crop(i, True) is not None]
        det_f, mc_f = [], []
        for i in valid:
            fd, fm = [], []
            for model, tf, lb in models:
                x = tf(Image.fromarray(get_crop(i, lb))).unsqueeze(0).to(device)
                with torch.no_grad():
                    fd.append(model(x).float().cpu().numpy()[0])
                    enable_mc_dropout(model)
                    fm.append(model(x.repeat(MC, 1, 1, 1)).float().cpu().numpy())
                    model.eval()
            det_f.append(np.stack(fd))
            mc_f.append(np.stack(fm))
        frames_out[f"det_{idx}"] = np.stack(det_f)
        frames_out[f"mc_{idx}"] = np.stack(mc_f)
        frames_out[f"center_{idx}"] = np.array(valid.index(center_idx))
        base_area = float(gt_mask.sum())
        meta.append({"index": int(idx), "case": case, "track_len": len(valid), "window_frames": len(nums),
                     "mean_area_ratio": float(np.mean([masks[i].sum() / base_area for i in valid])),
                     "seconds": time.time() - t0, "flow_seconds": t_flow})
        print(f"[{n + 1}/{len(mine)}] idx {idx} len {len(valid)}/{len(nums)} {meta[-1]['seconds']:.1f}s", flush=True)
        if (n + 1) % 25 == 0 or n + 1 == len(mine):
            np.savez_compressed(OUT_DIR / f"frames_shard{args.shard_id}.npz", **frames_out)
            (OUT_DIR / f"meta_shard{args.shard_id}.json").write_text(json.dumps(meta))


def frame_votes(det: np.ndarray, mc: np.ndarray, weights: np.ndarray, n_classes: int):
    onehot = mc.argmax(-1)[..., None] == np.arange(n_classes)
    v = np.einsum("m,fmc->fc", weights, onehot.mean(axis=2))
    v_det = np.einsum("m,fmc->fc", weights, (det.argmax(-1)[..., None] == np.arange(n_classes)).astype(float))
    return v, v_det


def score(args) -> None:
    members_cfg, weights, y, base, u, n_classes = ensemble_state()
    flagged = np.where(u >= GATE)[0]
    labels = list(range(n_classes))

    flow = {}
    for shard in (0, 1):
        path = OUT_DIR / f"frames_shard{shard}.npz"
        if path.exists():
            data = np.load(path)
            for key in data.files:
                if key.startswith("det_"):
                    i = int(key[4:])
                    flow[i] = (data[key], data[f"mc_{i}"], int(data[f"center_{i}"]))
    sam2 = {}
    for d in SAM2_DIRS:
        for shard in (0, 1):
            path = d / f"frames_shard{shard}.npz"
            if path.exists():
                data = np.load(path)
                for key in data.files:
                    if key.startswith("det_"):
                        i = int(key[4:])
                        sam2[i] = track_predictions(data[key], data[f"mc_{i}"], int(data[f"center_{i}"]), weights, n_classes)
    missing = [i for i in flagged if i not in flow]
    assert not missing, f"{len(missing)} flagged instances lack flow results"

    preds = {"flow majority of frame labels": base.copy(), "flow pooled vote": base.copy(),
             "flow uncertainty-weighted vote": base.copy(), "SAM2 uncertainty-weighted vote": base.copy()}
    for i in flagged:
        det, mc, center = flow[i]
        tb = np.einsum("m,fmc->c", weights, (det.argmax(-1)[..., None] == np.arange(n_classes)).astype(float))
        v, v_det = frame_votes(det, mc, weights, n_classes)
        frame_labels = plurality(v, v_det)
        counts = np.bincount(frame_labels, minlength=n_classes).astype(float)
        preds["flow majority of frame labels"][i] = int(plurality(counts[None], tb[None])[0])
        tp = track_predictions(det, mc, center, weights, n_classes)
        preds["flow pooled vote"][i] = tp[RULES[0]]
        preds["flow uncertainty-weighted vote"][i] = tp[RULES[1]]
        preds["SAM2 uncertainty-weighted vote"][i] = sam2[i][RULES[1]]

    out = {"n": int(len(y)), "flagged": int(len(flagged)),
           "ensemble_alone": {"accuracy": float((base == y).mean()), "macro_f1": float(f1_score(y, base, average="macro", labels=labels))}}
    print(f"ensemble alone: accuracy {out['ensemble_alone']['accuracy']:.4f}, macro-F1 {out['ensemble_alone']['macro_f1']:.4f}")
    ref_ok = preds["SAM2 uncertainty-weighted vote"] == y
    for name, p in preds.items():
        ok = p == y
        row = {"accuracy": float(ok.mean()), "macro_f1": float(f1_score(y, p, average="macro", labels=labels)),
               "fixed": int(((base != y) & ok).sum()), "broken": int(((base == y) & ~ok).sum())}
        if not name.startswith("SAM2"):
            a, b = int((ok & ~ref_ok).sum()), int((ref_ok & ~ok).sum())
            row.update({"only_this_correct_vs_sam2": a, "only_sam2_correct": b,
                        "mcnemar_p_vs_sam2": float(binomtest(a, a + b, 0.5).pvalue) if a + b else 1.0})
        out[name] = row
        extra = f", vs SAM2: only this {row.get('only_this_correct_vs_sam2')}, only SAM2 {row.get('only_sam2_correct')}, McNemar p {row.get('mcnemar_p_vs_sam2', float('nan')):.3f}" if "mcnemar_p_vs_sam2" in row else ""
        print(f"{name:<34} accuracy {row['accuracy']:.4f} macro-F1 {row['macro_f1']:.4f} fixed {row['fixed']} broken {row['broken']}{extra}")

    meta = []
    for shard in (0, 1):
        path = OUT_DIR / f"meta_shard{shard}.json"
        if path.exists():
            meta += json.loads(path.read_text())
    if meta:
        out["tracking"] = {"mean_track_len": float(np.mean([m["track_len"] for m in meta])),
                           "mean_window_frames": float(np.mean([m["window_frames"] for m in meta])),
                           "mean_area_ratio_vs_annotation": float(np.mean([m["mean_area_ratio"] for m in meta])),
                           "seconds_per_instance": float(np.mean([m["seconds"] for m in meta])),
                           "flow_seconds_per_instance": float(np.mean([m["flow_seconds"] for m in meta]))}
        print(out["tracking"])
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shard-id", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--device", default="cuda:1")
    r.add_argument("--data-root", type=Path, default=REPO_ROOT / "GraSP")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("score")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else score(args)


if __name__ == "__main__":
    main()
