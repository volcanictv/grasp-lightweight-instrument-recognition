"""Fine-tune SAM3 (Sam3TrackerModel, the SAM2-style interactive predictor of facebook/sam3) on GraSP with ground-truth boxes as prompts, the
same protocol as scripts/finetune_sam2_gtbox.py: train on one fold, choose the epoch on another by mean per-instrument mask IoU, never touch
the test cases. Trainable: mask decoder and prompt encoder always; --unfreeze-layers N also trains the FPN neck and the last N ViT layers.
The frozen part of the vision encoder is stored in fp16 (memory), the trained parts in fp32, all under fp16 autocast.

Usage (titanxp, sam3_venv):
    ~/sam3_venv/bin/python scripts/finetune_sam3_gtbox.py --out-dir experiments/sam3/dec --device cuda:0
    ~/sam3_venv/bin/python scripts/finetune_sam3_gtbox.py --out-dir experiments/sam3/enc4 --unfreeze-layers 4 --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from finetune_sam2_gtbox import instances, jitter, seg_loss
from surgical_ai.data.detection_dataset import GraspDetectionDataset

TRAINABLE_PREFIXES = ("mask_decoder", "prompt_encoder", "vision_encoder.neck", "vision_encoder.backbone.layers")


def predict(model, proc, image: Image.Image, boxes: np.ndarray, device: str, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask logits (n, 1, H, W) at the image size and predicted IoUs for n boxes of one image."""
    inputs = proc(images=image, input_boxes=[boxes.tolist()], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16):
        emb = model.get_image_embeddings(pixel_values)
    emb = [e.float() for e in emb]
    out = model(image_embeddings=emb, input_boxes=inputs["input_boxes"].to(device).float(), multimask_output=False)
    logits = proc.post_process_masks(out.pred_masks, inputs["original_sizes"], binarize=False)[0]
    return logits, out.iou_scores.reshape(-1)


def stats(values: list) -> dict:
    v = np.array(values)
    return {"mean_iou": float(v.mean()), "iou_ge_0.5": float((v >= 0.5).mean()), "iou_ge_0.75": float((v >= 0.75).mean()), "n": int(len(v))}


@torch.no_grad()
def evaluate(model, proc, ds, tta: bool, stride: int, device: str) -> dict:
    """Mean mask IoU over every stride-th frame; with tta, plain and flip-averaged scores from the same forward passes."""
    model.eval()
    plain, flip = [], []
    for file_name, anns in ds.samples[::stride]:
        boxes, masks = instances(ds, file_name, anns, None)
        if not len(boxes):
            continue
        image = Image.open(ds.frames_root / file_name).convert("RGB")
        logits, _ = predict(model, proc, image, boxes, device, grad=False)
        logits = logits.float()
        avg = None
        if tta:
            w = image.size[0]
            fboxes = boxes.copy()
            fboxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            flogits, _ = predict(model, proc, image.transpose(Image.FLIP_LEFT_RIGHT), fboxes, device, grad=False)
            avg = (logits + flogits.float().flip(-1)) / 2
        for j, gt in enumerate(masks):
            for lg, acc in ((logits[j, 0], plain), (avg[j, 0] if avg is not None else None, flip)):
                if lg is None:
                    continue
                m = (lg > 0).cpu().numpy()
                u = np.logical_or(m, gt).sum()
                acc.append(float(np.logical_and(m, gt).sum() / u) if u else 0.0)
    return {"plain": stats(plain), "flip_tta": stats(flip)} if tta else stats(plain)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-split", default="fold2")
    ap.add_argument("--dev-split", default="fold1")
    ap.add_argument("--model", default="facebook/sam3")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-layers", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.03)
    ap.add_argument("--max-instances", type=int, default=6)
    ap.add_argument("--dev-stride", type=int, default=3, help="per-epoch dev check on every k-th dev frame (the final check uses --final-stride)")
    ap.add_argument("--final-stride", type=int, default=1, help="final plain and flip check on every k-th dev frame (1 = all)")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap train frames")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    random.seed(0), np.random.seed(0), torch.manual_seed(0)

    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    model = Sam3TrackerModel.from_pretrained(args.model).to(args.device)
    proc = Sam3TrackerProcessor.from_pretrained(args.model)
    model.requires_grad_(False)
    model.vision_encoder.half()
    groups = [{"params": [*model.mask_decoder.parameters(), *model.prompt_encoder.parameters()], "lr": args.lr}]
    for p in groups[0]["params"]:
        p.requires_grad_(True)
    if args.unfreeze_layers:
        trained = [*model.vision_encoder.backbone.layers[-args.unfreeze_layers:], model.vision_encoder.neck]
        enc = []
        for part in trained:
            part.float()
            for p in part.parameters():
                p.requires_grad_(True)
                enc.append(p)
        groups.append({"params": enc, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    train_ds = GraspDetectionDataset(args.data_root, args.train_split, include_masks=True)
    dev_ds = GraspDetectionDataset(args.data_root, args.dev_split, include_masks=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    base = evaluate(model, proc, dev_ds, False, args.dev_stride, args.device)
    say(f"epoch 0 (zero-shot) dev/{args.dev_stride} {base}")
    log.append({"epoch": 0, **base})
    best, best_epoch = -1.0, 0
    n_frames = len(train_ds.samples[: args.limit])
    for epoch in range(1, args.epochs + 1):
        model.train()
        if not args.unfreeze_layers:
            model.vision_encoder.eval()
        order = list(range(n_frames))
        random.shuffle(order)
        total, n = 0.0, 0
        for i in order:
            file_name, anns = train_ds.samples[i]
            boxes, masks = instances(train_ds, file_name, anns, args.max_instances)
            if not len(boxes):
                continue
            image = Image.open(train_ds.frames_root / file_name).convert("RGB")
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = boxes.copy()
                boxes[:, [0, 2]] = image.size[0] - boxes[:, [2, 0]]
                masks = [m[:, ::-1].copy() for m in masks]
            logits, iou_pred = predict(model, proc, image, jitter(boxes, (image.size[1], image.size[0]), args.jitter), args.device, grad=bool(args.unfreeze_layers))
            gt = torch.from_numpy(np.stack(masks)).float().to(args.device)[:, None]
            logits = logits.float()
            with torch.no_grad():
                pos = (logits > 0).float()
                true_iou = (pos * gt).sum((1, 2, 3)) / ((pos + gt).clamp(max=1).sum((1, 2, 3)) + 1e-6)
            loss = seg_loss(logits, gt) + F.mse_loss(iou_pred.float(), true_iou)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            total, n = total + loss.item(), n + 1
        dev = evaluate(model, proc, dev_ds, False, args.dev_stride, args.device)
        say(f"epoch {epoch}/{args.epochs} train_loss {total / max(n, 1):.4f} dev/{args.dev_stride} {dev}")
        log.append({"epoch": epoch, "train_loss": total / max(n, 1), **dev})
        if dev["mean_iou"] > best:
            best, best_epoch = dev["mean_iou"], epoch
            torch.save({k: v for k, v in model.state_dict().items() if k.startswith(TRAINABLE_PREFIXES)}, args.out_dir / "weights.pt")
        (args.out_dir / "log.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "log": log}, indent=1))

    say(f"best epoch {best_epoch} dev/{args.dev_stride} mean IoU {best:.4f}")
    model.load_state_dict(torch.load(args.out_dir / "weights.pt", map_location=args.device), strict=False)
    res = evaluate(model, proc, dev_ds, True, args.final_stride, args.device)
    say(f"best checkpoint, dev frames 1/{args.final_stride}: {res}")
    (args.out_dir / "tta.json").write_text(json.dumps({"best_epoch": best_epoch, **res}, indent=1))


if __name__ == "__main__":
    main()
