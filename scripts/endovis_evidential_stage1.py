"""The evidential (Dirichlet) ensemble trained on an EndoVis dataset's own
sequences, the like-for-like counterpart of endovis_pipeline_stage1.py for the
MC-Dropout pipeline (docs/DECISIONS.md 2026-09-29, pre-registered).

Same instances, crops, member sizes and weights and training loop as the MC
stage 1; only the loss (EvidentialLoss, lambda 0.01, KL annealed over the first
5 of 10 epochs, evidence clamp 10, class weights from the training counts) and
the plain (non-dropout) architectures differ. Prediction is one deterministic
pass per member; nothing here uses MC passes or softmax.

Usage:
    python scripts/endovis_evidential_stage1.py --dataset 2018 --zip ~/Desktop/endovis_data/endovis2018.zip \
        --out-dir experiments_endovis/evid_stage1_2018 --device cuda:0
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import f1_score

import evaluate_endovis2017_generalization as e17
import evaluate_endovis2018_generalization as e18
import finetune_endovis_transfer as ft
from endovis_pipeline_stage1 import CropSet, crops_for
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.models import build_model
from surgical_ai.training.losses import EvidentialLoss

EPOCHS = 10
LAMBDA = 0.01
ANNEAL = 5
MEMBERS = [
    dict(label="resnet50_320", model="resnet50", size=320, letterbox=True, weight=0.40, batch=4),
    dict(label="resnet50_224", model="resnet50", size=224, letterbox=True, weight=0.20, batch=8),
    dict(label="baseline", model="mobilenet_v3_small", size=224, letterbox=False, weight=0.20, batch=8),
    dict(label="letterbox_crop", model="mobilenet_v3_small", size=224, letterbox=True, weight=0.20, batch=8),
]


def train_member(cfg, crops, y, device, seed):
    torch.manual_seed(seed)
    model = build_model(cfg["model"], num_classes=7, pretrained=True, freeze_backbone=False).to(device)
    counts = np.bincount(y, minlength=7).astype(float)
    w = np.where(counts > 0, counts.sum() / (7 * np.maximum(counts, 1)), 0.0)
    loss_fn = EvidentialLoss(LAMBDA, ANNEAL, class_weight=torch.tensor(w, dtype=torch.float32, device=device))
    head = [p for n, p in model.named_parameters() if n.startswith(("fc", "classifier"))]
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = torch.utils.data.DataLoader(CropSet(crops, y, cfg["size"], True), batch_size=32, shuffle=True,
                                         num_workers=2, drop_last=True)
    for ep in range(EPOCHS):
        model.train()
        loss_fn.set_epoch(ep + 1)
        tot = 0.0
        for x, t in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), t.to(device))
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        print(f"  {cfg['label']} epoch {ep + 1}/{EPOCHS} loss {tot / len(loader):.4f}", flush=True)
    return model


def single_pass_logits(model, crops, cfg, device):
    tf = build_transforms(cfg["size"], train=False)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), cfg["batch"]):
            x = torch.stack([tf(Image.fromarray(c)) for c in crops[i:i + cfg["batch"]]]).to(device)
            out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["2017", "2018"], required=True)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(args.zip)
    if args.dataset == "2018":
        tr = ft.instances_2018(zf, "train")
        raw_val = e18.extract_instances(zf, "val")
        va = [(img, m, b, name) for img, m, b, name, _src in raw_val]
        seq = np.array([Path(src).name.split("_frame")[0] for *_x, src in raw_val])
    else:
        tr, va = ft.instances_2017_train(zf), e17.extract_instances(zf)
        seq = np.array(["all"] * len(va))
    name_to_idx = {n: i for i, n in enumerate(ft.CLASS_NAMES)}
    y_tr = np.array([name_to_idx[n] for *_x, n in tr])
    y_va = np.array([name_to_idx[n] for *_x, n in va])
    crops = {(split, lb): crops_for(inst, lb) for split, inst in (("tr", tr), ("va", va)) for lb in (True, False)}
    del tr, va
    print(f"{args.dataset}: train {len(y_tr)} val {len(y_va)}", flush=True)

    extract = {"y": y_va, "seq": seq}
    alpha = 0.0
    for cfg in MEMBERS:
        model = train_member(cfg, crops[("tr", cfg["letterbox"])], y_tr, device, args.seed)
        torch.save(model.state_dict(), args.out_dir / f"{cfg['label']}.pt")
        logits = single_pass_logits(model, crops[("va", cfg["letterbox"])], cfg, device)
        extract[f"det_{cfg['label']}"] = logits
        alpha = alpha + cfg["weight"] * alpha_from_logits(logits)
        print(f"  {cfg['label']}: single-member accuracy {(logits.argmax(1) == y_va).mean():.4f}", flush=True)
        del model
        np.savez(args.out_dir / "extract.npz", **extract)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    pred = mu.argmax(axis=1)
    present = sorted(set(y_va.tolist()))
    s1 = variance_scores(alpha)["epistemic"]
    result = {"dataset": args.dataset, "seed": args.seed, "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
              "lambda": LAMBDA, "anneal_epochs": ANNEAL, "epochs": EPOCHS,
              "accuracy": float((pred == y_va).mean()),
              "macro_f1_present": float(f1_score(y_va, pred, labels=present, average="macro", zero_division=0)),
              "s1_quantiles": {str(q): float(np.quantile(s1, q)) for q in (0.5, 0.7, 0.8, 0.9, 0.95)}}
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
