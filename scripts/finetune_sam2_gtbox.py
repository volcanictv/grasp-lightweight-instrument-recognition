"""Fine-tune SAM2.1-large on GraSP with ground-truth boxes as prompts, selecting the checkpoint on a held-out dev fold (never on the
official test cases, unlike scripts/finetune_sam2_decoder.py). Scored by mean per-instrument mask IoU on the dev fold.

Trainable parts: the mask decoder and prompt encoder always; --unfreeze-blocks N also unfreezes the last N Hiera blocks and the
FPN neck. Box prompts are jittered a little during training. --tta averages the mask logits of the image and its horizontal flip.

Modes:
    --train-split fold2 --dev-split fold1   development run, picks the epoch count
    --train-split train --dev-split fold1   final run (fold1 is then inside the training data; only the fixed epoch count is used,
                                            pass --fixed-epochs N and the last epoch is saved)
Usage (titanxp, surgical environment):
    python scripts/finetune_sam2_gtbox.py --out-dir experiments/sam2_gtbox/dec --device cuda:0
    python scripts/finetune_sam2_gtbox.py --out-dir experiments/sam2_gtbox/enc4 --unfreeze-blocks 4 --device cuda:1
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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from surgical_ai.data.detection_dataset import GraspDetectionDataset
from surgical_ai.data.mask_utils import decode_instance_mask


def embed(predictor, image_np: np.ndarray, grad: bool) -> None:
    """set_image without its no_grad, so gradients reach the unfrozen encoder blocks."""
    model = predictor.model
    predictor._orig_hw = [image_np.shape[:2]]
    x = predictor._transforms(image_np)[None, ...].to(predictor.device)
    with torch.set_grad_enabled(grad):
        backbone_out = model.forward_image(x)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed
        feats = [f.permute(1, 2, 0).view(1, -1, *s) for f, s in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])][::-1]
    predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    predictor._is_image_set = True


def box_logits(predictor, boxes: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask logits (n, 1, H, W) and predicted IoUs for n boxes of the embedded image, gradient-preserving."""
    _mi, _uc, _l, unnorm_box = predictor._prep_prompts(None, None, boxes, None, True)
    coords = unnorm_box.reshape(-1, 2, 2)
    labels = torch.tensor([[2, 3]], dtype=torch.int, device=coords.device).repeat(coords.size(0), 1)
    sparse, dense = predictor.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    high_res = [f[-1].unsqueeze(0) for f in predictor._features["high_res_feats"]]
    low, iou_pred, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False, repeat_image=True, high_res_features=high_res)
    return predictor._transforms.postprocess_masks(low, predictor._orig_hw[-1]), iou_pred


def instances(ds: GraspDetectionDataset, file_name: str, anns: list, cap: int | None):
    boxes, masks = [], []
    for a in anns[:cap]:
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        m = decode_instance_mask(a["segmentation"]).astype(bool)
        if m.any():
            boxes.append([x, y, x + w, y + h])
            masks.append(m)
    return np.array(boxes, dtype=np.float32), masks


def seg_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * gt).sum(dim=(1, 2, 3))
    dice = 1.0 - (2 * inter + 1.0) / (prob.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) + 1.0)
    return F.binary_cross_entropy_with_logits(logits, gt) + dice.mean()


def jitter(boxes: np.ndarray, hw: tuple[int, int], frac: float) -> np.ndarray:
    out = boxes.copy()
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]] * 2, axis=1)
    out += np.random.uniform(-frac, frac, boxes.shape).astype(np.float32) * wh
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, hw[1] - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, hw[0] - 1)
    return out


@torch.no_grad()
def evaluate(predictor, ds, tta: bool, limit: int | None) -> dict:
    predictor.model.eval()
    ious = []
    for file_name, anns in ds.samples[:limit]:
        boxes, masks = instances(ds, file_name, anns, None)
        if not len(boxes):
            continue
        img = np.array(Image.open(ds.frames_root / file_name).convert("RGB"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embed(predictor, img, grad=False)
            logits, _ = box_logits(predictor, boxes)
            if tta:
                w = img.shape[1]
                embed(predictor, img[:, ::-1].copy(), grad=False)
                fboxes = boxes.copy()
                fboxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                flogits, _ = box_logits(predictor, fboxes)
                logits = (logits.float() + flogits.float().flip(-1)) / 2
        for lg, gt in zip(logits[:, 0], masks):
            m = (lg > 0).cpu().numpy()
            u = np.logical_or(m, gt).sum()
            ious.append(float(np.logical_and(m, gt).sum() / u) if u else 0.0)
    ious = np.array(ious)
    return {"mean_iou": float(ious.mean()), "iou_ge_0.5": float((ious >= 0.5).mean()), "iou_ge_0.75": float((ious >= 0.75).mean()), "n": int(len(ious))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-split", default="fold2")
    ap.add_argument("--dev-split", default="fold1")
    ap.add_argument("--sam-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fixed-epochs", type=int, default=None, help="train exactly this many epochs and save the last (final run)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.03, help="box jitter as a fraction of box size")
    ap.add_argument("--max-instances", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap train and dev frames")
    ap.add_argument("--tta", action="store_true", help="also report horizontal-flip averaged dev IoU for the best checkpoint")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="training seed (0 reproduces every earlier run)")
    args = ap.parse_args()
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(args.sam_config, str(args.sam_checkpoint), device=args.device))
    model = predictor.model
    model.requires_grad_(False)
    groups = [{"params": [*model.sam_mask_decoder.parameters(), *model.sam_prompt_encoder.parameters()], "lr": args.lr}]
    for p in groups[0]["params"]:
        p.requires_grad_(True)
    if args.unfreeze_blocks:
        blocks = list(model.image_encoder.trunk.blocks)[-args.unfreeze_blocks:]
        enc = [p for part in (*blocks, model.image_encoder.neck) for p in part.parameters()]
        for p in enc:
            p.requires_grad_(True)
        groups.append({"params": enc, "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    train_ds = GraspDetectionDataset(args.data_root, args.train_split, include_masks=True)
    dev_ds = GraspDetectionDataset(args.data_root, args.dev_split, include_masks=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    base = evaluate(predictor, dev_ds, False, args.limit)
    say(f"epoch 0 (zero-shot) dev {base}")
    log.append({"epoch": 0, **base})
    best, best_epoch = -1.0, 0
    epochs = args.fixed_epochs or args.epochs
    for epoch in range(1, epochs + 1):
        model.sam_mask_decoder.train()
        model.sam_prompt_encoder.eval()
        model.image_encoder.train(bool(args.unfreeze_blocks))
        order = list(range(len(train_ds.samples[: args.limit])))
        random.shuffle(order)
        total, n = 0.0, 0
        for i in order:
            file_name, anns = train_ds.samples[i]
            boxes, masks = instances(train_ds, file_name, anns, args.max_instances)
            if not len(boxes):
                continue
            img = np.array(Image.open(train_ds.frames_root / file_name).convert("RGB"))
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                boxes = boxes.copy()
                boxes[:, [0, 2]] = img.shape[1] - boxes[:, [2, 0]]
                masks = [m[:, ::-1].copy() for m in masks]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embed(predictor, img, grad=bool(args.unfreeze_blocks))
                logits, iou_pred = box_logits(predictor, jitter(boxes, img.shape[:2], args.jitter))
            gt = torch.from_numpy(np.stack(masks)).float().to(args.device)[:, None]
            logits = logits.float()
            with torch.no_grad():
                true_iou = ((logits > 0).float() * gt).sum((1, 2, 3)) / (((logits > 0).float() + gt).clamp(max=1).sum((1, 2, 3)) + 1e-6)
            loss = seg_loss(logits, gt) + F.mse_loss(iou_pred.float()[:, 0], true_iou)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            total, n = total + loss.item(), n + 1
        dev = evaluate(predictor, dev_ds, False, args.limit)
        say(f"epoch {epoch}/{epochs} train_loss {total / max(n, 1):.4f} dev {dev}")
        log.append({"epoch": epoch, "train_loss": total / max(n, 1), **dev})
        trainable = {k: v for k, v in model.state_dict().items() if k.startswith(("sam_mask_decoder", "sam_prompt_encoder", "image_encoder.neck", "image_encoder.trunk.blocks"))}
        if args.fixed_epochs:
            torch.save(trainable, args.out_dir / "weights.pt")
        elif dev["mean_iou"] > best:
            best, best_epoch = dev["mean_iou"], epoch
            torch.save(trainable, args.out_dir / "weights.pt")
        (args.out_dir / "log.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "log": log}, indent=1))

    say(f"best epoch {best_epoch} dev mean IoU {best:.4f}" if not args.fixed_epochs else f"saved epoch {epochs}")
    if args.tta:
        model.load_state_dict(torch.load(args.out_dir / "weights.pt", map_location=args.device), strict=False)
        model.eval()
        res = {"plain": evaluate(predictor, dev_ds, False, args.limit), "flip_tta": evaluate(predictor, dev_ds, True, args.limit)}
        say(f"best checkpoint: {res}")
        (args.out_dir / "tta.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
