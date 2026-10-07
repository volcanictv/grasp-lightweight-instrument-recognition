"""GFLOPs of every stage of the pipeline, counted with torch.utils.flop_counter.FlopCounterMode (matrix multiplications, convolutions and attention) on one forward pass of the real architectures.
The counter reports 2 FLOPs per multiply-add; we divide by two so that the figures follow the multiply-add convention of fvcore, the counter behind most published FLOP tables.
Weights do not change a FLOP count, so the SAM3 model is built from its configuration (random weights).

  --part main   (surgical environment) SAM2.1-large and -tiny image encoder + box decoder, the four classifier members, SAM2-large propagation over the 21 frames of one instrument
  --part sam3   (sam3 environment)     SAM3 image encoder + box decoder

Usage (titanxp, repo root, a GPU with a few GB free):
    CUDA_VISIBLE_DEVICES=1 python scripts/count_gflops.py --part main --out docs/reports/gflops_main.json
    CUDA_VISIBLE_DEVICES=1 ~/sam3_venv/bin/python scripts/count_gflops.py --part sam3 --out docs/reports/gflops_sam3.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

CKPT = Path(os.environ.get("SAM2_CKPT_DIR", Path.home() / "sam2" / "checkpoints"))
DATA = Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP"))
dev = "cuda"


def gflops(fn) -> float:
    fc = FlopCounterMode(display=False)
    with fc:
        fn()
    return fc.get_total_flops() / 2 / 1e9  # multiply-adds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["main", "sam3"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res: dict = {"convention": "GFLOPs, one multiply-add counted as one FLOP (fvcore convention); matmul, convolution and attention only"}
    frame = np.array(Image.open(DATA / "frames-001" / "frames" / "CASE041" / "00035.jpg").convert("RGB"))
    h, w = frame.shape[:2]
    boxes1 = np.array([[100, 100, 400, 360]], dtype=np.float32)
    boxes3 = np.array([[100, 100, 400, 360], [500, 200, 800, 500], [900, 100, 1200, 300]], dtype=np.float32)

    if args.part == "sam3":
        import finetune_sam3_gtbox as s3
        from transformers import AutoConfig, Sam3TrackerModel, Sam3TrackerProcessor
        model = Sam3TrackerModel(AutoConfig.from_pretrained("facebook/sam3")).to(dev).eval().requires_grad_(False)
        proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        image = Image.fromarray(frame)
        with torch.no_grad():
            s3.predict(model, proc, image, boxes1, dev, grad=False)  # warm-up
            g1 = gflops(lambda: s3.predict(model, proc, image, boxes1, dev, grad=False))
            g3 = gflops(lambda: s3.predict(model, proc, image, boxes3, dev, grad=False))
        res["sam3"] = {"one_pass_one_box": g1, "one_pass_three_boxes": g3, "per_extra_box": (g3 - g1) / 2, "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
        print(res)
        args.out.write_text(json.dumps(res, indent=1))
        return

    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from finetune_sam2_gtbox import box_logits, embed
    from surgical_ai.models import build_model

    for name, (ck, cfg) in {"sam2_large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"), "sam2_tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml")}.items():
        pred = SAM2ImagePredictor(build_sam2(cfg, str(CKPT / ck), device=dev))
        pred.model.eval()

        def run(b):
            with torch.inference_mode():
                embed(pred, frame, grad=False)
                box_logits(pred, b)

        run(boxes1)
        g1, g3 = gflops(lambda: run(boxes1)), gflops(lambda: run(boxes3))
        res[name] = {"one_pass_one_box": g1, "one_pass_three_boxes": g3, "per_extra_box": (g3 - g1) / 2, "params_m": sum(p.numel() for p in pred.model.parameters()) / 1e6}
        del pred
        torch.cuda.empty_cache()

    cfg = yaml.safe_load((REPO_ROOT / "configs/rc_ens4_N_s44.yaml").read_text())
    cls = {}
    for m in cfg["members"]:
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(dev).eval()
        x = torch.randn(1, 3, m["image_size"], m["image_size"], device=dev)
        with torch.no_grad():
            net(x)
            cls[m["label"]] = {"gflops": gflops(lambda: net(x)), "params_m": sum(p.numel() for p in net.parameters()) / 1e6, "image_size": m["image_size"]}
    res["classifier_members"] = cls
    res["classifier_four_members"] = sum(v["gflops"] for v in cls.values())

    # SAM2-large propagation over the 21 frames (keyframe, 10 forward, 10 backward) of one instrument
    tmp = Path(os.environ.get("BENCH_TMP", "/tmp/gflops_frames"))
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    nums = list(range(25, 46))
    for k, n in enumerate(nums):
        shutil.copy(DATA / "frames-001" / "frames" / "CASE041" / f"{n:05d}.jpg", tmp / f"{k:05d}.jpg")
    predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", str(CKPT / "sam2.1_hiera_large.pt"), device=dev)
    mask = np.zeros((h, w), bool)
    mask[120:340, 120:380] = True

    def track():
        with torch.inference_mode():
            state = predictor.init_state(video_path=str(tmp))
            predictor.add_new_mask(state, frame_idx=10, obj_id=1, mask=mask)
            for _ in predictor.propagate_in_video(state):
                pass
            for _ in predictor.propagate_in_video(state, reverse=True):
                pass

    track()
    res["sam2_large_tracking_21_frames"] = gflops(track)
    res["refined_instrument"] = res["sam2_large_tracking_21_frames"] + 21 * res["classifier_four_members"]
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(res, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
