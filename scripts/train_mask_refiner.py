"""Trains the mask-refinement network (src/surgical_ai/models/mask_refiner.py) on crops dumped by scripts/dump_refiner_crops.py.
The training crops must come from a SAM that did not train on those frames (cross-fit), so the refiner sees the error profile it
will meet on unseen frames. The epoch is chosen on the dev crops by mean mask IoU in crop space.

Usage (titanxp, surgical environment):
    python scripts/train_mask_refiner.py --train experiments/refiner/fold2_encB.npz --dev experiments/refiner/fold1_enc4.npz --out-dir experiments/refiner/run1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from surgical_ai.models.mask_refiner import LOGIT_CLIP, MaskRefiner, box_channel


def load(path: Path, device: str) -> dict:
    z = np.load(path)
    return {"img": torch.from_numpy(z["img"]), "logit": torch.from_numpy(z["logit"]), "gt": torch.from_numpy(z["gt"]),
            "boxes": torch.from_numpy(z["boxes"]), "geom": torch.from_numpy(z["geom"]), "size": int(z["size"])}


def batch(d: dict, ids: np.ndarray, device: str) -> tuple[torch.Tensor, ...]:
    img = d["img"][ids].to(device).permute(0, 3, 1, 2).float() / 255
    logit = (d["logit"][ids].to(device).float()[:, None] / 255) * 2 * LOGIT_CLIP - LOGIT_CLIP
    gt = d["gt"][ids].to(device).float()[:, None] / 255
    box = box_channel(d["boxes"][ids].to(device), d["geom"][ids].to(device), d["size"])
    return img, logit, box, gt


def seg_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * gt).sum(dim=(1, 2, 3))
    dice = 1.0 - (2 * inter + 1.0) / (prob.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) + 1.0)
    return F.binary_cross_entropy_with_logits(logits, gt) + dice.mean()


@torch.no_grad()
def evaluate(model: MaskRefiner, d: dict, device: str, bs: int = 32) -> dict:
    model.eval()
    n = len(d["img"])
    base, ref = [], []
    for i in range(0, n, bs):
        ids = np.arange(i, min(i + bs, n))
        img, logit, box, gt = batch(d, ids, device)
        out = model(img, logit, box)
        g = gt > 0.5
        for pred, acc in (((logit > 0) & (box > 0.5), base), (out > 0, ref)):
            inter = (pred & g).sum(dim=(1, 2, 3)).float()
            union = (pred | g).sum(dim=(1, 2, 3)).float()
            acc.extend((inter / union.clamp(min=1)).tolist())
    return {"sam_iou": float(np.mean(base)), "refined_iou": float(np.mean(ref)), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train, dev = load(args.train, args.device), load(args.dev, args.device)
    model = MaskRefiner(pretrained=not args.no_pretrained).to(args.device)
    enc = [*model.stem.parameters(), *model.l1.parameters(), *model.l2.parameters(), *model.l3.parameters(), *model.l4.parameters()]
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.encoder_lr}, {"params": rest, "lr": args.lr}], weight_decay=1e-4)
    steps = args.epochs * ((len(train["img"]) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.encoder_lr, args.lr], total_steps=steps, pct_start=0.1)

    log = []
    base = evaluate(model, dev, args.device)
    print(f"[{time.strftime('%H:%M:%S')}] epoch 0 dev {base}", flush=True)
    best = base["refined_iou"]
    torch.save(model.state_dict(), args.out_dir / "weights.pt")  # the identity network, kept if no epoch beats SAM
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train["img"]))
        total, n = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            ids = np.sort(order[i:i + args.batch_size])
            img, logit, box, gt = batch(train, ids, args.device)
            if random.random() < 0.5:
                img, logit, box, gt = (t.flip(-1) for t in (img, logit, box, gt))
            if random.random() < 0.5:
                img, logit, box, gt = (t.flip(-2) for t in (img, logit, box, gt))
            out = model(img, logit, box)
            loss = seg_loss(out, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        dev_res = evaluate(model, dev, args.device)
        print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch}/{args.epochs} train_loss {total / n:.4f} dev {dev_res}", flush=True)
        log.append({"epoch": epoch, "train_loss": total / n, **dev_res})
        if dev_res["refined_iou"] > best:
            best = dev_res["refined_iou"]
            torch.save(model.state_dict(), args.out_dir / "weights.pt")
        (args.out_dir / "log.json").write_text(json.dumps({"baseline": base, "log": log}, indent=1))
    print(f"best dev refined IoU {best:.4f} (SAM {base['sam_iou']:.4f})")


if __name__ == "__main__":
    main()
