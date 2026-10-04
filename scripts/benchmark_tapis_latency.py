"""Per-keyframe latency of TAPIS (BCV-Uniandes/GraSP, TAPIS/) for instrument segmentation and classification on this machine, with the shipped
weights of the official split: the Swin-L Mask2Former region-proposal network (pretrained_models/train/SEGMENTATION_BASELINE/swinl.pth) on one full
frame, then the MViT video model (pretrained_models/train/INSTRUMENTS.pyth) on a 16-frame clip with the proposed regions. The two stages are timed
separately and as a sum, batch size 1, after warm-up, with CUDA synchronisation. The clip tensor holds a real test frame repeated 16 times (the 30 fps
source frames are not available here; latency does not depend on pixel values). Regions are the Mask2Former detections of the frame (at most 5, the
repository's MAX_BBOXES), with their query features.

Usage (titanxp, tapis_venv, an idle GPU):
    CUDA_VISIBLE_DEVICES=0 ~/tapis_venv/bin/python scripts/benchmark_tapis_latency.py --reps 50 --out docs/reports/latency/tapis.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

TAPIS_ROOT = Path.home() / "baselines" / "GraSP" / "TAPIS"
sys.path.insert(0, str(TAPIS_ROOT))

import numpy as np
import torch
from PIL import Image


def stats(v: list[float]) -> dict:
    v = sorted(v)
    return {"mean_ms": 1000 * statistics.mean(v), "median_ms": 1000 * statistics.median(v), "p95_ms": 1000 * v[int(0.95 * (len(v) - 1))], "n": len(v)}


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=TAPIS_ROOT / "data" / "GraSP")
    ap.add_argument("--rpn-config", default="region_proposals/configs/grasp/GraSP_SwinL_regions.yaml")
    ap.add_argument("--rpn-weights", type=Path, default=None)
    ap.add_argument("--video-config", default="configs/GraSP/TAPIS/TAPIS_INSTRUMENTS.yaml")
    ap.add_argument("--video-weights", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rpn_weights = args.rpn_weights or args.data / "pretrained_models" / "train" / "SEGMENTATION_BASELINE" / "swinl.pth"
    video_weights = args.video_weights or args.data / "pretrained_models" / "train" / "INSTRUMENTS.pyth"
    frame_path = args.frame
    if frame_path is None:
        frame_path = next((Path.home() / "Desktop" / "Classification Surgurical Tools" / "GraSP" / "frames-001" / "frames").rglob("*.jpg"))
    frame = np.array(Image.open(frame_path).convert("RGB"))
    print("frame", frame_path, frame.shape, flush=True)

    # --- stage 1: Mask2Former region proposals (the repository's own class and configuration)
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from region_proposals import MaskFormer
    from region_proposals.mask2former import add_maskformer2_config
    rpn_cfg = get_cfg()
    add_deeplab_config(rpn_cfg)
    add_maskformer2_config(rpn_cfg)
    rpn_cfg.merge_from_file(str(TAPIS_ROOT / args.rpn_config))
    rpn_cfg.MODEL.WEIGHTS = str(rpn_weights)
    rpn_cfg.MODEL.DEVICE = "cuda"
    rpn = MaskFormer(rpn_cfg).to("cuda").eval()
    DetectionCheckpointer(rpn).load(str(rpn_weights))
    n_rpn = sum(p.numel() for p in rpn.parameters()) / 1e6

    img = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).float().cuda()  # BGR, as the detectron2 pipeline feeds it

    @torch.no_grad()
    def run_rpn():
        return rpn([img])

    # --- stage 2: MViT video model on the clip with regions (the repository's own model builder)
    from tapis.config.defaults import assert_and_infer_cfg, get_cfg as get_tapis_cfg
    from tapis.models import build_model
    from tapis.utils.checkpoint import load_checkpoint
    cfg = get_tapis_cfg()
    cfg.merge_from_file(str(TAPIS_ROOT / args.video_config))
    cfg.NUM_GPUS = 1
    cfg.FEATURES.ENABLE = True
    cfg.FEATURES.USE_RPN = False
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg = assert_and_infer_cfg(cfg)
    video = build_model(cfg).cuda().eval()
    load_checkpoint(str(video_weights), video, data_parallel=False, convert_from_caffe2=False)
    n_video = sum(p.numel() for p in video.parameters()) / 1e6
    T, S, B = cfg.DATA.NUM_FRAMES, cfg.DATA.TEST_CROP_SIZE, cfg.DATA.MAX_BBOXES
    clip = torch.from_numpy(np.array(Image.fromarray(frame).resize((S, S)))).permute(2, 0, 1).float().div(255).cuda()
    clip = clip[None, :, None].repeat(1, 1, T, 1, 1)
    boxes = torch.tensor([[20, 20, 200, 200]] * B, dtype=torch.float32).cuda()[None]
    feats = torch.randn(1, B, cfg.FEATURES.DIM_FEATURES).cuda()
    mask = torch.ones(1, B, dtype=torch.bool).cuda()

    @torch.no_grad()
    def run_video():
        return video([clip], bboxes=boxes, features=feats, boxes_mask=mask, images=None)

    for _ in range(args.warmup):
        run_rpn()
        run_video()
    t_rpn, t_video, t_total = [], [], []
    for _ in range(args.reps):
        t0 = sync()
        run_rpn()
        t1 = sync()
        run_video()
        t2 = sync()
        t_rpn.append(t1 - t0)
        t_video.append(t2 - t1)
        t_total.append(t2 - t0)
    res = {"gpu": torch.cuda.get_device_name(0), "frame": str(frame_path), "frame_hw": list(frame.shape[:2]), "clip": [T, S, S], "regions": B,
           "params_m": {"mask2former_swinl": n_rpn, "mvit_tapis": n_video}, "mask2former": stats(t_rpn), "video_model": stats(t_video), "total": stats(t_total)}
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
