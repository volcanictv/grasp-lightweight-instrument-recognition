"""Logits of a fine-tuned DINO classifier (scripts/endovis_dino_train.py) on the frames of the SAM2 tracks of the EndoVis validation instruments (scripts/endovis_gtbox_track.py), for the revisit.

Same crops as scripts/endovis_gtbox_logits.py (masked, letterboxed, the instrument box at the keyframe and the box of the tracked mask elsewhere). Output: npz with det_<instrument index> (frames x 7) and center_<index>.

    HF_HOME=$HOME/dino_baseline/hf PYTHONPATH=$HOME/dino_baseline/pylib python scripts/endovis_dino_extract.py --backbone dinov2-base --weights <model.pt> \
        --tracks tracks_all/tracks_shard0.pkl tracks_all/tracks_shard1.pkl --out dino_tracks.npz
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
from torchvision import transforms as T

import endovis_gtbox_common as C
from endovis_dino_train import MEAN, STD, Net


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, choices=["dinov2-base", "dinov3-base"])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tracks", type=Path, nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    net = Net(args.backbone).to(dev)
    net.load_state_dict(torch.load(args.weights, map_location=dev))
    net.eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    insts = C.load_instances(args.split)
    tracks: dict[int, dict] = {}
    for path in args.tracks:
        tracks.update(pickle.loads(path.read_bytes()))
    out: dict[str, np.ndarray] = {}
    for n, (idx, frames) in enumerate(sorted(tracks.items())):
        box = insts[idx]["box"]
        crops, valid = [], []
        for off in sorted(frames):
            num, rle = frames[off]
            m = C.decode(rle)
            if off != 0 and m.sum() < C.MIN_AREA:
                continue
            b = box if off == 0 else C.mask_box(m)
            crops.append(C.masked_crop(C.load_frame(insts[idx]["seq"], num), m, b, True))
            valid.append(off)
        with torch.no_grad():
            x = torch.stack([tf(Image.fromarray(c)) for c in crops]).to(dev)
            out[f"det_{idx}"] = net(x).float().cpu().numpy()
        out[f"center_{idx}"] = np.array(valid.index(0))
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(tracks)} tracks", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
