"""Does the recipe transfer? Fine-tune the ResNet-50 (320 px, letterbox) region
classifier on an EndoVis dataset's own training sequences and score it on that
dataset's held-out sequences, starting either from GraSP weights or from
ImageNet only. This tests the architecture and recipe on another dataset; it is
not a zero-shot generalization number and must not be reported as one.

Splits are sequence-disjoint by construction: the releases ship train/ and
val/ (2018) or train/ and val1..val10 (2017) as separate sequences. The number
of epochs is fixed and the last epoch is used, so nothing is selected on the
held-out sequences. The evaluation instances are the same ones the zero-shot
evaluation scores.

Usage:
    python scripts/finetune_endovis_transfer.py --dataset 2018 --zip ~/Desktop/endovis_data/endovis2018.zip \
        --init grasp --seed 42 --out experiments_endovis/2018_grasp_s42.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn

import evaluate_endovis2017_generalization as e17
import evaluate_endovis2018_generalization as e18
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

CLASS_NAMES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
               "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
GRASP_CKPT = "experiments/region_letterbox_resnet50_320_deepdropout_20260919-155832/best.pt"
IMAGE_SIZE = 320
EPOCHS = 10


def instances_2018(zf: zipfile.ZipFile, split: str):
    return [(img, m, b, name) for img, m, b, name, _src in e18.extract_instances(zf, split)]


def instances_2017_train(zf: zipfile.ZipFile):
    """Same extraction as the val folds in e17, pointed at train/."""
    names = set(zf.namelist())
    out = []
    for label_name in sorted(n for n in names if "/train/label/" in n and n.endswith(".bmp")):
        image_name = label_name.replace("/label/", "/image/")
        if image_name not in names:
            continue
        label = np.array(Image.open(io.BytesIO(zf.read(label_name))))
        image = np.array(Image.open(io.BytesIO(zf.read(image_name))).convert("RGB"))
        for endovis_id, grasp_name in e17.ENDOVIS_ID_TO_GRASP_NAME.items():
            class_mask = label == endovis_id
            if not class_mask.any():
                continue
            labeled, n = e17.ndimage.label(class_mask)
            for c in range(1, n + 1):
                comp = labeled == c
                if comp.sum() < e17.MIN_COMPONENT_AREA:
                    continue
                ys, xs = np.nonzero(comp)
                out.append((image, comp, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), grasp_name))
    return out


def to_crops(instances):
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    crops = [e18.crop_instance(img, m, b, True) for img, m, b, _n in instances]
    y = np.array([name_to_idx[n] for *_x, n in instances])
    return crops, y


class CropSet(torch.utils.data.Dataset):
    def __init__(self, crops, y, train: bool):
        self.crops, self.y = crops, y
        self.tf = build_transforms(IMAGE_SIZE, train=train, augmentation="default")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.crops[i])), int(self.y[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["2017", "2018"], required=True)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--init", choices=["grasp", "imagenet"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    zf = zipfile.ZipFile(args.zip)
    if args.dataset == "2018":
        train_inst, val_inst = instances_2018(zf, "train"), instances_2018(zf, "val")
    else:
        train_inst, val_inst = instances_2017_train(zf), e17.extract_instances(zf)
    tr_x, tr_y = to_crops(train_inst)
    va_x, va_y = to_crops(val_inst)
    del train_inst, val_inst
    print(f"{args.dataset} train {len(tr_y)} val {len(va_y)} init={args.init} seed={args.seed}", flush=True)

    model = build_model("resnet50_deepdropout", num_classes=7, pretrained=args.init == "imagenet",
                        freeze_backbone=False).to(device)
    if args.init == "grasp":
        model.load_state_dict(torch.load(REPO_ROOT / GRASP_CKPT, map_location=device), strict=False)

    counts = np.bincount(tr_y, minlength=7).astype(float)
    w = np.where(counts > 0, counts.sum() / (7 * np.maximum(counts, 1)), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    head = [p for n, p in model.named_parameters() if n.startswith(("fc", "classifier"))]
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = torch.utils.data.DataLoader(CropSet(tr_x, tr_y, True), batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        print(f"epoch {ep + 1}/{EPOCHS} loss {tot / len(loader):.4f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _y in torch.utils.data.DataLoader(CropSet(va_x, va_y, False), batch_size=64, num_workers=4):
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    present = sorted(set(va_y.tolist()))
    p, r, f1, sup = precision_recall_fscore_support(va_y, pred, labels=list(range(7)), zero_division=0)
    result = {
        "dataset": args.dataset, "init": args.init, "seed": args.seed, "epochs": EPOCHS,
        "n_train": int(len(tr_y)), "n_val": int(len(va_y)),
        "accuracy": float(accuracy_score(va_y, pred)),
        "macro_f1_present": float(f1_score(va_y, pred, labels=present, average="macro", zero_division=0)),
        "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(7)},
        "support": {CLASS_NAMES[i]: int(sup[i]) for i in range(7)},
        "confusion_counts": confusion_matrix(va_y, pred, labels=list(range(7))).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("accuracy", "macro_f1_present", "n_train", "n_val")}))


if __name__ == "__main__":
    main()
