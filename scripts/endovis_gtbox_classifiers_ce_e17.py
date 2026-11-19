"""Cross-entropy deep-dropout baseline members on EndoVis 2018 plus EndoVis 2017, for the softmax and MC-dropout gate baselines.

Exactly the recipe of endovis_gtbox_classifiers_grasp_e17.py (same crops, 2017 extra data, tracker-style neighbour crops with probability 0.5, Adam lr 1e-3 head / 1e-4 backbone, batch 32, 20 epochs,
class weights from the training counts, last epoch) with only two changes, as for the GraSP baselines (scripts/make_ce_official_configs.py): the loss is class-weighted cross-entropy and the networks are the
deep-dropout variants that MC dropout needs.

    python scripts/endovis_gtbox_classifiers_ce_e17.py --seed 42 --neighbours <dir> --zip2017 <endovis2017.zip> --out-dir <dir> [--only resnet50_224 ...]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch

import endovis_evidential_stage1 as es1
import endovis_gtbox_classifiers_grasp as G
import endovis_gtbox_classifiers_grasp_e17 as E
import endovis_gtbox_common as C
from surgical_ai.models import build_model

DROPOUT_MODEL = {"resnet50": "resnet50_deepdropout", "mobilenet_v3_small": "mobilenet_v3_small_deepdropout"}


def train_member_ce(cfg, crops, y, options, device, seed, log):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = build_model(DROPOUT_MODEL[cfg["model"]], num_classes=7, pretrained=True, freeze_backbone=False).to(device)
    counts = np.bincount(y, minlength=7).astype(float)
    w = np.where(counts > 0, counts.sum() / (7 * np.maximum(counts, 1)), 0.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    head = [p for n, p in model.named_parameters() if n.startswith(("fc", "classifier"))]
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.Adam([{"params": body, "lr": 1e-4}, {"params": head, "lr": 1e-3}])
    ds = G.NeighbourCropSet(crops, y, options, cfg["size"], cfg["letterbox"], G.NEIGHBOUR_PROB)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
    for ep in range(G.EPOCHS):
        model.train()
        tot = 0.0
        for x, t in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), t.to(device))
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  {cfg['label']} epoch {ep + 1}/{G.EPOCHS} loss {tot / len(loader):.4f}", flush=True)
        log.append({"member": cfg["label"], "epoch": ep + 1, "loss": tot / len(loader)})
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--neighbours", type=Path, required=True)
    ap.add_argument("--zip2017", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    insts = C.load_instances("train")
    y18 = np.array([i["cls"] for i in insts])
    meta = json.loads((args.neighbours / "meta.json").read_text())
    options = {int(k): [args.neighbours / "crops" / it["file"] for it in v] for k, v in meta.items() if int(k) < len(insts)}
    crops = {lb: [] for lb in (True, False)}
    cache: dict[tuple[str, int], np.ndarray] = {}
    for inst in insts:
        key = (inst["seq"], inst["frame"])
        if key not in cache:
            cache.clear()
            cache[key] = C.load_frame(*key)
        mask = C.decode(inst["rle"])
        for lb in crops:
            crops[lb].append(C.masked_crop(cache[key], mask, inst["box"], lb))
    c17, y17 = E.crops_2017(args.zip2017)
    for lb in crops:
        crops[lb] += c17[lb]
    y = np.concatenate([y18, y17])
    print(f"seed {args.seed}: {len(y18)} 2018 + {len(y17)} 2017 train instances, per class {np.bincount(y, minlength=7).tolist()}", flush=True)
    log: list[dict] = []
    for cfg in es1.MEMBERS:
        if args.only and cfg["label"] not in args.only:
            continue
        model = train_member_ce(cfg, crops[cfg["letterbox"]], y, options, torch.device(args.device), args.seed, log)
        torch.save(model.state_dict(), args.out_dir / f"{cfg['label']}.pt")
        del model
        torch.cuda.empty_cache()
    tag = "_".join(args.only) if args.only else "all"
    (args.out_dir / f"train_log_{tag}.json").write_text(json.dumps({"loss": "class-weighted cross-entropy", "networks": "deep-dropout variants", "epochs": G.EPOCHS, "log": log}))


if __name__ == "__main__":
    main()
