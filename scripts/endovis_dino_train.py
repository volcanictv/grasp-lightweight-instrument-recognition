"""DINOv2 / DINOv3 single-frame reference classifiers on EndoVis 2018 + 2017 crops (annotation masks), for the ensemble ablation.

The recipe of the GraSP reference rows (~/dino_baseline/train_dino.py, fine-tuning): ViT-B backbone, concat(CLS, mean patch token) -> linear head, AdamW with layer-wise learning-rate decay 0.75 (base 2e-5, head 1e-3,
weight decay 0.05), one warm-up epoch then cosine, batch 16 x 2 accumulation, 15 epochs, class-weighted cross-entropy, the default augmentation at 224 px; the members' crops (masked, letterboxed) and training data
(EndoVis 2018 train + the 2017 release as in scripts/endovis_gtbox_classifiers_grasp_e17.py), no tracker-style neighbour crops (a single-frame model). One run, no tuning. Output: logits on the validation
instruments (annotation masks) for the scorer.

    HF_HOME=$HOME/dino_baseline/hf PYTHONPATH=$HOME/dino_baseline/pylib python scripts/endovis_dino_train.py --backbone dinov2-base --seed 42 --zip2017 <endovis2017.zip> --masks <masks_ens.pkl> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

import endovis_gtbox_classifiers_grasp_e17 as E
import endovis_gtbox_common as C
from surgical_ai.data.transforms import build_transforms
from surgical_ai.training.losses import EvidentialLoss, compute_class_weights

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Backbone(nn.Module):
    """concat(CLS, mean patch token) of the final normalised tokens."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name.startswith("dinov2"):
            from transformers import Dinov2Model
            self.m = Dinov2Model.from_pretrained(f"facebook/{name}", attn_implementation="sdpa")
            self.dim, self.depth = self.m.config.hidden_size, self.m.config.num_hidden_layers
        else:
            import timm
            self.m = timm.create_model("hf-hub:timm/vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
            self.dim, self.depth = self.m.num_features, len(self.m.blocks)

    def forward(self, x):
        if self.name.startswith("dinov2"):
            h = self.m(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        h = self.m.forward_features(x)
        return torch.cat([h[:, 0], h[:, self.m.num_prefix_tokens:].mean(1)], 1)

    def layer_id(self, pname: str) -> int:
        if self.name.startswith("dinov2"):
            if pname.startswith("m.embeddings"):
                return 0
            mm = re.match(r"m\.encoder\.layer\.(\d+)\.", pname)
            return int(mm.group(1)) + 1 if mm else self.depth + 1
        mm = re.match(r"m\.blocks\.(\d+)\.", pname)
        if mm:
            return int(mm.group(1)) + 1
        return self.depth + 1 if pname.startswith("m.norm") or pname.startswith("m.fc_norm") else 0


class Net(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.backbone = Backbone(name)
        self.head = nn.Linear(2 * self.backbone.dim, 7)

    def forward(self, x):
        return self.head(self.backbone(x))


class CropSet(torch.utils.data.Dataset):
    def __init__(self, crops, y, tf):
        self.crops, self.y, self.tf = crops, y, tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.crops[i])), int(self.y[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=["dinov2-base", "dinov3-base"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip2017", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True, help="annotation masks of the validation instruments (masks_ens.pkl)")
    ap.add_argument("--loss", choices=["ce", "evidential"], default="ce", help="evidential: the Dirichlet loss of the paper's members (lambda 0.01 annealed over 10 epochs), weights are saved")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: time and memory of this many batches, then stop")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    insts = C.load_instances("train")
    crops, cache = [], {}
    for inst in insts:
        key = (inst["seq"], inst["frame"])
        if key not in cache:
            cache.clear()
            cache[key] = C.load_frame(*key)
        crops.append(C.masked_crop(cache[key], C.decode(inst["rle"]), inst["box"], True))
    c17, y17 = E.crops_2017(args.zip2017)
    crops += c17[True]
    y = np.concatenate([np.array([i["cls"] for i in insts]), y17])
    cw = compute_class_weights(torch.bincount(torch.from_numpy(y), minlength=7).float()).to(dev)
    print(f"train {len(y)} crops, per class {np.bincount(y, minlength=7).tolist()}, class weights {[round(float(v), 3) for v in cw]}", flush=True)

    vinsts = C.load_instances("val")
    vmasks = pickle.loads(args.masks.read_bytes())
    vcrops, vkey, vframe = [], None, None
    for inst in vinsts:
        key = (inst["seq"], inst["frame"])
        if key != vkey:
            vkey, vframe = key, C.load_frame(*key)
        vcrops.append(C.masked_crop(vframe, C.decode(vmasks[inst["i"]]), inst["box"], True))
    yv = np.array([i["cls"] for i in vinsts])

    net = Net(args.backbone).to(dev)
    L, decay, base_lr, head_lr, wd = net.backbone.depth, 0.75, 2e-5, 1e-3, 0.05
    groups: dict = {}
    for n, p in net.named_parameters():
        if n.startswith("head"):
            lr, lid = head_lr, "head"
        else:
            lid = net.backbone.layer_id(n[len("backbone."):])
            lr = base_lr * decay ** (L + 1 - lid)
        nodecay = p.ndim < 2 or n.endswith(".bias") or any(k in n for k in ("cls_token", "pos", "mask_token", "storage", "reg_token"))
        groups.setdefault((lid, nodecay), dict(params=[], lr=lr, weight_decay=0.0 if nodecay else wd))["params"].append(p)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(list(groups.values()))
    dl = torch.utils.data.DataLoader(CropSet(crops, y, build_transforms(224, True, "default")), batch_size=args.bs, shuffle=True, num_workers=3, drop_last=True, pin_memory=True)
    steps_per_epoch = len(dl) // args.accum
    total, warm = steps_per_epoch * args.epochs, steps_per_epoch
    print(f"optimizer groups {len(groups)}, steps/epoch {steps_per_epoch}", flush=True)
    step, log = 0, []
    t0 = time.time()
    loss_fn = EvidentialLoss(0.01, 10, class_weight=cw) if args.loss == "evidential" else None
    tag = "" if args.loss == "ce" else "_evidential"
    for ep in range(args.epochs):
        net.train()
        if loss_fn is not None:
            loss_fn.set_epoch(ep + 1)
        te, run, nb = time.time(), 0.0, 0
        for b, (xb, yb) in enumerate(dl):
            if b % args.accum == 0 and step < total:
                f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
            out = net(xb.to(dev, non_blocking=True))
            loss = loss_fn(out, yb.to(dev)) if loss_fn is not None else F.cross_entropy(out, yb.to(dev), weight=cw)
            (loss / args.accum).backward()
            run += float(loss)
            nb += 1
            if (b + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if args.max_batches and b + 1 >= args.max_batches:
                torch.cuda.synchronize()
                print(f"smoke {b + 1} batches {time.time() - te:.1f}s peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; full run would be about {(time.time() - te) / (b + 1) * len(dl) * args.epochs / 60:.0f} min", flush=True)
                return
        log.append({"epoch": ep + 1, "train_loss": run / nb, "seconds": time.time() - te})
        print(log[-1], flush=True)
    net.eval()
    evaltf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    outs = []
    with torch.no_grad():
        for i in range(0, len(vcrops), 64):
            xb = torch.stack([evaltf(Image.fromarray(c)) for c in vcrops[i:i + 64]]).to(dev)
            outs.append(net(xb).float().cpu().numpy())
    logits = np.concatenate(outs)
    np.save(args.out_dir / f"val_logits_{args.backbone}{tag}_s{args.seed}.npy", logits)
    if args.loss == "evidential":
        torch.save(net.state_dict(), args.out_dir / f"model_{args.backbone}{tag}_s{args.seed}.pt")
    pred = logits.argmax(1)
    res = {"backbone": args.backbone, "loss": args.loss, "seed": args.seed, "val_accuracy": float((pred == yv).mean()), "wall_seconds": time.time() - t0, "epoch_log": log}
    (args.out_dir / f"result_{args.backbone}{tag}_s{args.seed}.json").write_text(json.dumps(res, indent=1))
    print(f"validation accuracy {100 * res['val_accuracy']:.2f}  wall {res['wall_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
