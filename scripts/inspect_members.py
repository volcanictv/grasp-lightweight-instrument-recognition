"""What does the recognition ensemble look at? Per-instance inspection of chosen test instruments (ground-truth masks, official seed-42 members).

For each instrument: the masked crop the classifier sees, Grad-CAM of the first member (ResNet-50, 320 px) for the true and the predicted class, the ensemble probabilities of the original crop and of three reduced
versions (silhouette only, scrambled texture inside the mask, blurred) to tell shape evidence from texture evidence, and the pooled features and logits for a later neuron-level analysis.

    python scripts/inspect_members.py --sets /tmp/insp_sets.json --masks experiments/gtbox_sam/gtmask/masks.pkl --out /tmp/insp
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import yaml
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFilter
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import crop_from_box
from evidential_seeds_e2e_eval import CONFIGS
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.evidential import alpha_from_logits
from surgical_ai.models import build_model

CLASSES = ["BF", "PF", "LND", "MCS", "SI", "CA", "LG"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/arms/ens4_N_official_s42.yaml")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    spec = json.loads(args.sets.read_text())
    masks = pickle.loads(args.masks.read_bytes())
    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    cfg = yaml.safe_load(args.config.read_text())
    weights = CONFIGS["four"]
    members = []
    for m in cfg["members"]:
        net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(dev)
        net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=dev), strict=False)
        members.append((m["label"], net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))
    resnet = members[0][1]
    head = [mod for mod in resnet.fc.modules() if isinstance(mod, torch.nn.Linear)][-1] if not isinstance(resnet.fc, torch.nn.Linear) else resnet.fc
    frames_root = args.data_root / "frames-001" / "frames"

    def ensemble(crops_by_member: dict) -> np.ndarray:
        alpha = 0
        for label, net, tf, _lb in members:
            with torch.no_grad():
                lg = net(tf(Image.fromarray(crops_by_member[label])).unsqueeze(0).to(dev)).float().cpu().numpy()
            alpha = alpha + weights[label] * alpha_from_logits(lg)
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        return mu[0]

    def gradcam(x: torch.Tensor, cls: int) -> np.ndarray:
        store: dict = {}
        def fwd_hook(_m, _i, o):
            store["a"] = o
            o.register_hook(lambda g: store.__setitem__("g", g))

        h = resnet.layer4.register_forward_hook(fwd_hook)
        resnet.zero_grad()
        out = resnet(x)
        out[0, cls].backward()
        h.remove()
        w = store["g"].mean((2, 3), keepdim=True)
        cam = torch.relu((w * store["a"]).sum(1))[0].detach().cpu().numpy()
        return cam / (cam.max() + 1e-8)

    def overlay(img: Image.Image, cam: np.ndarray, size: int = 170) -> Image.Image:
        base = img.convert("RGB").resize((size, size))
        heat = Image.fromarray((cm.jet(np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0)[..., :3] * 255).astype(np.uint8))
        return Image.blend(base, heat, 0.45)

    rows, feats, logit_store, order = [], [], [], []
    tiles = []
    for set_name, idxs in spec["sets"].items():
        for idx in idxs:
            file_name, _seg, box, label = ds.instances[idx]
            frame = np.array(Image.open(frames_root / file_name).convert("RGB"))
            mask = mask_codec.decode(masks[idx]).astype(bool)
            crops = {lab: crop_from_box(frame, mask, box, lb) for lab, _n, _t, lb in members}
            base = crops["resnet50_320"]
            p0 = ensemble(crops)
            # each reduction is applied to every member's own crop (same operation per member)
            def reduce(fn):
                return ensemble({lab: fn(c) for lab, c in crops.items()})
            def f_sil(c):
                o = c.copy(); o[c.any(axis=2)] = 127; return o
            def f_shuf(c):
                o = c.copy(); ins_ = c.any(axis=2); v = c[ins_]; o[ins_] = v[np.random.default_rng(0).permutation(len(v))]; return o
            def f_blur(c):
                return np.array(Image.fromarray(c).filter(ImageFilter.GaussianBlur(4)))
            variants = {"orig": p0, "silhouette": reduce(f_sil), "texture_scrambled": reduce(f_shuf), "blur4": reduce(f_blur)}
            pred = int(p0.argmax())
            x = members[0][2](Image.fromarray(base)).unsqueeze(0).to(dev)
            with torch.no_grad():
                feat_hook = []
                hh = resnet.avgpool.register_forward_hook(lambda m, i, o: feat_hook.append(o.flatten(1).cpu().numpy()[0]))
                lg0 = resnet(x).float().cpu().numpy()[0]
                hh.remove()
            feats.append(feat_hook[0]); logit_store.append(lg0); order.append(idx)
            cam_true = gradcam(x.clone().requires_grad_(False), int(label))
            cam_pred = gradcam(x.clone().requires_grad_(False), pred)
            x0, y0, w, h = [int(round(v)) for v in box]
            pad = int(0.15 * max(w, h))
            ctx = Image.fromarray(frame[max(0, y0 - pad): y0 + h + pad, max(0, x0 - pad): x0 + w + pad]).convert("RGB").resize((170, 170))
            tiles.append((set_name, idx, spec["meta"][str(idx)], ctx, Image.fromarray(base).convert("RGB").resize((170, 170)), overlay(Image.fromarray(base), cam_true), overlay(Image.fromarray(base), cam_pred), pred, variants))
            rows.append({"set": set_name, "idx": idx, **spec["meta"][str(idx)], "pred_ens": CLASSES[pred], "variants": {k: [round(float(v), 3) for v in p] for k, p in variants.items()}})
            print(set_name, idx, spec["meta"][str(idx)]["case"], CLASSES[label], "->", CLASSES[pred], {k: CLASSES[int(p.argmax())] for k, p in variants.items()}, flush=True)

    W = 170
    sheet = Image.new("RGB", (4 * W, len(tiles) * (W + 16)), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for r, (set_name, idx, meta, ctx, crop, ct, cp, pred, variants) in enumerate(tiles):
        y0 = r * (W + 16)
        d.text((4, y0 + 1), f"{set_name[:2]} #{idx} {meta['case']} true {meta['true']} ens {CLASSES[pred]} TAPIS {meta['tapis']} | P_LG orig {variants['orig'][6]:.2f} sil {variants['silhouette'][6]:.2f} scr {variants['texture_scrambled'][6]:.2f} blur {variants['blur4'][6]:.2f}", fill=(255, 255, 0))
        for c, im in enumerate((ctx, crop, ct, cp)):
            sheet.paste(im, (c * W, y0 + 14))
    sheet.save(args.out / "sheet.png")
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    np.savez(args.out / "feats.npz", idx=np.array(order), feat=np.array(feats), logits=np.array(logit_store), head_w=head.weight.detach().cpu().numpy(), head_b=head.bias.detach().cpu().numpy())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
