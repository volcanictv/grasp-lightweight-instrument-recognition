"""Logits of an ensemble config on a held-out target, for the evidential validation
(docs/DECISIONS.md 2026-09-21). Always saves the deterministic pass per member; saves
`mc_samples` dropout-on passes per member only when asked (arm C). Targets: the held-out
GraSP fold (grasp_fold1 / grasp_fold2) or EndoVis val (endovis2018 / endovis2017, zero-shot,
same instance extraction and crop recipe as evaluate_endovis_vote_ensemble.py).

Usage:
    python scripts/evidential_extract.py --ensemble-config configs/evidential/ens_E_fold1_s42_lam0p1a10.yaml \
        --target grasp_fold1 --data-root GraSP --out experiments_edl/extract/E_grasp_fold1_s42.npz
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image

import evaluate_endovis2017_generalization as e17
import evaluate_endovis2018_generalization as e18
from analyze_uncertainty_signals import enable_mc_dropout
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

CLASS_NAMES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
               "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def run_member(model, loader_batches, device, mc_samples):
    det, mc = [], [[] for _ in range(mc_samples)]
    with torch.no_grad():
        for x in loader_batches:
            x = x.to(device)
            model.eval()
            det.append(model(x).float().cpu().numpy())
            if mc_samples:
                enable_mc_dropout(model)
                for s in range(mc_samples):
                    mc[s].append(model(x).float().cpu().numpy())
                model.eval()
    out = {"det": np.concatenate(det)}
    if mc_samples:
        out["mc"] = np.stack([np.concatenate(p) for p in mc])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-config", type=Path, required=True)
    ap.add_argument("--target", choices=["grasp_fold1", "grasp_fold2", "endovis2018", "endovis2017"], required=True)
    ap.add_argument("--data-root", type=Path, default=Path("GraSP"))
    ap.add_argument("--zip", type=Path, default=None)
    ap.add_argument("--mc-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = yaml.safe_load(args.ensemble_config.read_text())
    members = cfg["members"]

    instances = None
    if args.target.startswith("endovis"):
        zf = zipfile.ZipFile(args.zip)
        if args.target == "endovis2018":
            instances = [(img, m, b, name) for img, m, b, name, _s in e18.extract_instances(zf, "val")]
        else:
            instances = e17.extract_instances(zf)
        name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
        y = np.array([name_to_idx[n] for *_x, n in instances])
        case = np.array(["endovis"] * len(y))
    result = {}
    for m in members:
        model = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
        tf = build_transforms(m["image_size"], train=False)
        if instances is None:
            ds = GraspRegionDataset(args.data_root, args.target.replace("grasp_", ""), transform=tf, letterbox=m["letterbox"])
            y_m = np.array([lbl for _fn, _seg, _box, lbl in ds.instances])
            case_m = np.array([fn.split("/")[0] for fn, _s, _b, _l in ds.instances])
            assert "y" not in result or (result["y"] == y_m).all()
            result["y"], result["case"] = y_m, case_m
            loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
            batches = (imgs for imgs, _l in loader)
        else:
            crops = [e18.crop_instance(img, mk, bb, m["letterbox"]) for img, mk, bb, _n in instances]
            bs = 4 if m["image_size"] >= 320 else 8

            def gen(crops=crops, tf=tf, bs=bs):
                for i in range(0, len(crops), bs):
                    yield torch.stack([tf(Image.fromarray(c)) for c in crops[i:i + bs]])
            batches = gen()
            result["y"], result["case"] = y, case
        out = run_member(model, batches, device, args.mc_samples)
        result["det_" + m["label"]] = out["det"]
        if args.mc_samples:
            result["mc_" + m["label"]] = out["mc"]
        print(m["label"], out["det"].shape, flush=True)
        del model
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
