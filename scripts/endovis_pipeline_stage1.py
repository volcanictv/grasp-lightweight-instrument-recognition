"""The full classification architecture on an EndoVis dataset, without tracking:
train the four ensemble members on the dataset's own training sequences (same
recipe as on GraSP, ImageNet start), then take the weighted plurality of 20
dropout-on passes per member, and apply the vote-disagreement gate at the 9%
threshold chosen on GraSP, unchanged. Reports accuracy and macro-F1 of the
ensemble, the share of instances the gate would send to tracking, and accuracy
on the flagged versus unflagged instances. Per-instance votes and softmax
confidences are saved so later analysis (tracking, softmax comparison) needs no
retraining.

Usage:
    python scripts/endovis_pipeline_stage1.py --dataset 2018 --zip ~/Desktop/endovis_data/endovis2018.zip --out-dir experiments_endovis/stage1_2018
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
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

import evaluate_endovis2017_generalization as e17
import evaluate_endovis2018_generalization as e18
import finetune_endovis_transfer as ft
from analyze_uncertainty_signals import enable_mc_dropout
from build_tracking_sample_b import plurality
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

GATE = 0.09
MC = 20
EPOCHS = 10
MEMBERS = [
    dict(label="resnet50_320", model="resnet50_deepdropout", size=320, letterbox=True, weight=0.40, batch=4),
    dict(label="resnet50_224", model="resnet50_deepdropout", size=224, letterbox=True, weight=0.20, batch=8),
    dict(label="baseline", model="mobilenet_v3_small_deepdropout", size=224, letterbox=False, weight=0.20, batch=8),
    dict(label="letterbox_crop", model="mobilenet_v3_small_deepdropout", size=224, letterbox=True, weight=0.20, batch=8),
]


class CropSet(torch.utils.data.Dataset):
    def __init__(self, crops, y, size, train):
        self.crops, self.y, self.tf = crops, y, build_transforms(size, train=train, augmentation="default")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.crops[i])), int(self.y[i])


def crops_for(instances, letterbox):
    return [e18.crop_instance(img, m, b, letterbox) for img, m, b, _n in instances]


def train_member(cfg, crops, y, device, seed):
    torch.manual_seed(seed)
    model = build_model(cfg["model"], num_classes=7, pretrained=True, freeze_backbone=False).to(device)
    counts = np.bincount(y, minlength=7).astype(float)
    w = np.where(counts > 0, counts.sum() / (7 * np.maximum(counts, 1)), 0.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    head = [p for n, p in model.named_parameters() if n.startswith(("fc", "classifier"))]
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loader = torch.utils.data.DataLoader(CropSet(crops, y, cfg["size"], True), batch_size=32, shuffle=True,
                                         num_workers=4, drop_last=True)
    for ep in range(EPOCHS):
        model.train()
        for x, t in loader:
            opt.zero_grad()
            loss_fn(model(x.to(device)), t.to(device)).backward()
            opt.step()
        sched.step()
    return model


def member_outputs(model, crops, cfg, device):
    tf = build_transforms(cfg["size"], train=False)
    mc_share, det_prob = [], []
    with torch.no_grad():
        for i in range(0, len(crops), cfg["batch"]):
            x = torch.stack([tf(Image.fromarray(c)) for c in crops[i:i + cfg["batch"]]]).to(device)
            model.eval()
            det_prob.append(torch.softmax(model(x), 1).cpu().numpy())
            enable_mc_dropout(model)
            logits = model(x.repeat_interleave(MC, dim=0)).view(len(x), MC, -1)
            model.eval()
            mc_share.append((logits.argmax(2)[..., None].cpu().numpy() == np.arange(7)).mean(axis=1))
    return np.concatenate(mc_share), np.concatenate(det_prob)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["2017", "2018"], required=True)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    zf = zipfile.ZipFile(args.zip)
    if args.dataset == "2018":
        tr, va = ft.instances_2018(zf, "train"), ft.instances_2018(zf, "val")
    else:
        tr, va = ft.instances_2017_train(zf), e17.extract_instances(zf)
    name_to_idx = {n: i for i, n in enumerate(ft.CLASS_NAMES)}
    y_tr = np.array([name_to_idx[n] for *_x, n in tr])
    y_va = np.array([name_to_idx[n] for *_x, n in va])
    crops = {(split, lb): crops_for(inst, lb) for split, inst in (("tr", tr), ("va", va)) for lb in (True, False)}
    del tr, va
    print(f"{args.dataset}: train {len(y_tr)} val {len(y_va)}", flush=True)

    v_mc = np.zeros((len(y_va), 7))
    v_det = np.zeros((len(y_va), 7))
    p_mean = np.zeros((len(y_va), 7))
    for cfg in MEMBERS:
        model = train_member(cfg, crops[("tr", cfg["letterbox"])], y_tr, device, args.seed)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out_dir / f"{cfg['label']}.pt")
        mc, det_p = member_outputs(model, crops[("va", cfg["letterbox"])], cfg, device)
        det_vote = (det_p.argmax(1)[:, None] == np.arange(7)).astype(float)
        v_mc += cfg["weight"] * mc
        v_det += cfg["weight"] * det_vote
        p_mean += cfg["weight"] * det_p
        print(f"  {cfg['label']}: single-member accuracy {(det_p.argmax(1) == y_va).mean():.4f}", flush=True)
        del model
    pred = plurality(v_mc, v_det)
    u = 1.0 - v_mc.max(axis=1)
    flag = u >= GATE
    present = sorted(set(y_va.tolist()))
    acc = (pred == y_va)
    result = {
        "dataset": args.dataset, "seed": args.seed, "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
        "gate": GATE, "accuracy": float(acc.mean()),
        "macro_f1_present": float(f1_score(y_va, pred, labels=present, average="macro", zero_division=0)),
        "share_flagged": float(flag.mean()), "n_flagged": int(flag.sum()),
        "accuracy_flagged": float(acc[flag].mean()) if flag.any() else None,
        "accuracy_unflagged": float(acc[~flag].mean()),
        "errors": int((~acc).sum()), "errors_flagged": int((~acc & flag).sum()),
        "confusion_counts": confusion_matrix(y_va, pred, labels=list(range(7))).tolist(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=1))
    np.savez(args.out_dir / "votes.npz", y=y_va, pred=pred, u=u, v_mc=v_mc, v_det=v_det, p_mean=p_mean)
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_counts"}, indent=1))


if __name__ == "__main__":
    main()
