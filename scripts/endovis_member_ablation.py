"""Single-member comparison on the EndoVis 2018 ISINet split: does adding EndoVis 2017 to the training set help the weak classes?

Variants (the only thing that changes is the training set; recipe of endovis_evidential_stage1.train_member, ResNet-50 224, letterboxed masked
crops of the ground-truth instances, evidential loss, 10 epochs, last-epoch weights):
    2018            ISINet's training sequences only
    2018+2017       plus every EndoVis 2017 frame; 2017 types BF, PF, LND, MCS keep their class, "Other" (type 7) becomes Ultrasound Probe,
                    Vessel Sealer and Grasping Retractor are dropped (no 2018 counterpart)
    2018+2017-noup  the same without the "Other" instances, to separate the effect of more BF/PF/LND/MCS data from the extra UP data
Scored on the untouched ISINet validation sequences (2, 5, 9, 15), which are never trained on.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_codec
from scipy import ndimage
from sklearn.metrics import f1_score

from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model
from surgical_ai.training.losses import EvidentialLoss

NAMES = ["BF", "PF", "LND", "MCS", "UP", "SI", "CA"]
T2017 = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}  # 2017 type value -> 2018 class index; 4 (vessel sealer) and 5 (grasping retractor) are dropped
MIN_AREA = 500
EPOCHS, LAMBDA, ANNEAL = 10, 0.01, 5


def pad_square(crop: np.ndarray) -> np.ndarray:
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    sq = np.zeros((side, side, 3), np.uint8)
    top, left = (side - ch) // 2, (side - cw) // 2
    sq[top:top + ch, left:left + cw] = crop
    return sq


def crop_of(img: np.ndarray, mask: np.ndarray, box: tuple) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in box)
    return pad_square((img[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8))


def crops_2018(root: Path, split: str) -> tuple[list, np.ndarray, list]:
    table = pickle.loads((root / "instances.pkl").read_bytes())[split]
    crops, y, seqs, cache = [], [], [], {}
    for inst in table:
        key = (inst["seq"], inst["frame"])
        if key not in cache:
            cache.clear()
            cache[key] = np.array(Image.open(root / "frames" / f"{key[0]}_frame{key[1]:03d}.jpg").convert("RGB"))
        crops.append(crop_of(cache[key], mask_codec.decode(inst["rle"]).astype(bool), inst["box"]))
        y.append(inst["cls"])
        seqs.append(inst["seq"])
    return crops, np.array(y), seqs


def crops_2017(zip_path: Path, cache_file: Path) -> tuple[list, np.ndarray, list]:
    if cache_file.exists():
        d = pickle.loads(cache_file.read_bytes())
        return d["crops"], d["y"], d["seq"]
    zf = zipfile.ZipFile(zip_path)
    names = set(zf.namelist())
    labels = [n for n in names if "/label/" in n and n.endswith(".bmp")]
    # the release ships labels for all 300 frames of sequence 9 but the image of frame 299 only
    skipped = {n for n in labels if n.replace("/label/", "/image/") not in names}
    print(f"2017: skipping {len(skipped)} labels without an image", flush=True)
    labels = [n for n in labels if n not in skipped]
    crops, y, seq = [], [], []
    for n in sorted(labels):
        m = re.search(r"seq_(\d+)_frame(\d+)", n)
        lab = np.array(Image.open(io.BytesIO(zf.read(n))))
        lab = lab[..., 0] if lab.ndim == 3 else lab
        img = np.array(Image.open(io.BytesIO(zf.read(n.replace("/label/", "/image/")))).convert("RGB"))
        for v, cls in T2017.items():
            comp, k = ndimage.label(lab == v)
            for j in range(1, k + 1):
                mask = comp == j
                if mask.sum() < MIN_AREA:
                    continue
                ys, xs = np.nonzero(mask)
                crops.append(crop_of(img, mask, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
                y.append(cls)
                seq.append(f"e17_seq{m.group(1)}")
    cache_file.write_bytes(pickle.dumps({"crops": crops, "y": np.array(y), "seq": seq}))
    return crops, np.array(y), seq


class CropSet(torch.utils.data.Dataset):
    def __init__(self, crops, y, size, train):
        self.crops, self.y, self.tf = crops, y, build_transforms(size, train=train, augmentation="default")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.crops[i])), int(self.y[i])


def train_member(crops, y, device, seed, size=224, batch=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model("resnet50", num_classes=7, pretrained=True, freeze_backbone=False).to(device)
    counts = np.bincount(y, minlength=7).astype(float)
    w = np.where(counts > 0, counts.sum() / (7 * np.maximum(counts, 1)), 0.0)
    loss_fn = EvidentialLoss(LAMBDA, ANNEAL, class_weight=torch.tensor(w, dtype=torch.float32, device=device))
    head = [p for n, p in model.named_parameters() if n.startswith(("fc", "classifier"))]
    hid = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = torch.utils.data.DataLoader(CropSet(crops, y, size, True), batch_size=batch, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    for ep in range(EPOCHS):
        model.train()
        loss_fn.set_epoch(ep + 1)
        tot = 0.0
        for x, t in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device, non_blocking=True)), t.to(device))
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        print(f"  epoch {ep + 1}/{EPOCHS} loss {tot / len(loader):.4f}", flush=True)
    return model


def predict(model, crops, device, size=224):
    tf = build_transforms(size, train=False)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            x = torch.stack([tf(Image.fromarray(c)) for c in crops[i:i + 64]]).to(device)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="EndoVis 2018 cache (frames/ and instances.pkl)")
    ap.add_argument("--zip2017", type=Path, required=True)
    ap.add_argument("--variant", choices=["2018", "2018+2017", "2018+2017-noup"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    work = args.out.parent
    work.mkdir(parents=True, exist_ok=True)
    tr_c, tr_y, tr_seq = crops_2018(args.root, "train")
    if args.variant != "2018":
        c17, y17, s17 = crops_2017(args.zip2017, work / "e17_crops.pkl")
        keep = np.ones(len(y17), bool) if args.variant == "2018+2017" else y17 != 4
        tr_c = tr_c + [c for c, k in zip(c17, keep) if k]
        tr_y = np.concatenate([tr_y, y17[keep]])
    va_c, va_y, _ = crops_2018(args.root, "val")
    print(f"{args.variant} seed {args.seed}: train {len(tr_y)} per class {np.bincount(tr_y, minlength=7).tolist()}, val {len(va_y)}", flush=True)
    dev = torch.device(args.device)
    model = train_member(tr_c, tr_y, dev, args.seed)
    logits = predict(model, va_c, dev)
    alpha = np.exp(np.clip(logits, -10, 10)) + 1.0
    prob = alpha / alpha.sum(1, keepdims=True)
    pred = prob.argmax(1)
    rec = [float(((pred == c) & (va_y == c)).sum() / max(1, (va_y == c).sum())) for c in range(7)]
    prec = [float(((pred == c) & (va_y == c)).sum() / max(1, (pred == c).sum())) for c in range(7)]
    f1 = f1_score(va_y, pred, average=None, labels=range(7), zero_division=0)
    res = {"variant": args.variant, "seed": args.seed, "train_per_class": np.bincount(tr_y, minlength=7).tolist(), "acc": float((pred == va_y).mean()),
           "macro_f1": float(f1.mean()), "per_class_f1": dict(zip(NAMES, map(float, f1))), "recall": dict(zip(NAMES, rec)), "precision": dict(zip(NAMES, prec)),
           "auroc_UP": auroc(prob[:, 4], va_y == 4), "n_pred_UP": int((pred == 4).sum())}
    args.out.write_text(json.dumps(res, indent=1))
    np.savez(args.out.with_suffix(".npz"), logits=logits, y=va_y)
    print(json.dumps({k: v for k, v in res.items() if k in ("acc", "macro_f1", "recall", "precision", "auroc_UP", "n_pred_UP")}), flush=True)


if __name__ == "__main__":
    main()
