"""Single-pass and MC-dropout outputs of the cross-entropy deep-dropout members (scripts/endovis_gtbox_classifiers_ce_e17.py) on the EndoVis validation crops (annotation masks).

For every validation instrument and member: the deterministic logits (det_<member>) and 20 stochastic passes with dropout active and batch norm in eval mode (mc_<member>, 80 passes over the four members),
as scripts/baselines_on_test.py does for GraSP. Only the keyframe crops are needed: the gate signals order the instruments, the revisit itself uses the evidential ensemble's tracks.

    python scripts/endovis_ce_extract.py --members experiments/endovis_isinet_e17/ce_members/s42 --masks experiments/endovis_isinet_e17/masks/masks_ens.pkl --out ce_val_s42.npz
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image

import endovis_evidential_stage1 as es1
import endovis_gtbox_common as C
from endovis_gtbox_classifiers_ce_e17 import DROPOUT_MODEL
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

MC_PASSES = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    by_label = {cfg["label"]: cfg for cfg in es1.MEMBERS}
    nets = {}
    for label in C.ORDER:
        cfg = by_label[label]
        net = build_model(DROPOUT_MODEL[cfg["model"]], num_classes=7, pretrained=False, freeze_backbone=False).to(dev)
        net.load_state_dict(torch.load(args.members / f"{label}.pt", map_location=dev))
        nets[label] = (cfg, net.eval(), build_transforms(cfg["size"], train=False))
    insts = C.load_instances(args.split)
    masks = pickle.loads(args.masks.read_bytes())
    crops = {lb: [] for lb in (True, False)}
    cache_key, frame = None, None
    for inst in insts:
        key = (inst["seq"], inst["frame"])
        if key != cache_key:
            cache_key, frame = key, C.load_frame(*key)
        m = C.decode(masks[inst["i"]])
        for lb in crops:
            crops[lb].append(C.masked_crop(frame, m, inst["box"], lb))

    def dropout_on(net) -> None:
        for mod in net.modules():
            if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                mod.train()

    out: dict[str, np.ndarray] = {"y": np.array([i["cls"] for i in insts])}
    for label, (cfg, net, tf) in nets.items():
        det, mc = [], []
        step = cfg["batch"] * 2
        for i in range(0, len(insts), step):
            x = torch.stack([tf(Image.fromarray(c)) for c in crops[cfg["letterbox"]][i:i + step]]).to(dev)
            with torch.no_grad():
                net.eval()
                det.append(net(x).float().cpu().numpy())
                dropout_on(net)
                mc.append(np.stack([net(x).float().cpu().numpy() for _ in range(MC_PASSES)]))
                net.eval()
        out[f"det_{label}"] = np.concatenate(det)
        out[f"mc_{label}"] = np.concatenate(mc, axis=1)
        print(f"{label}: single-pass accuracy {(out[f'det_{label}'].argmax(1) == out['y']).mean():.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
