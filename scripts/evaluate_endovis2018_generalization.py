"""Generalization test (priority 2, CLAUDE.md accuracy-first realignment):
run the GraSP-trained Task B region classifier, unmodified and never
trained on this data, against EndoVis 2018's instrument-type segmentation
masks -- a second, independent dataset from a different capture setup.

Class overlap confirmed against a primary source
(github.com/BCV-Uniandes/ISINet, data/robotseg_to_coco.py -- the same lab
that built GraSP) rather than a secondhand summary: EndoVis 2018's 2018
instrument-type task uses categories 1-7 = Bipolar Forceps, Prograsp
Forceps, Large Needle Driver, Monopolar Curved Scissors, Ultrasound Probe,
Suction Instrument, Clip Applier (0=background). 6 of these 7 match GraSP's
7 classes exactly by name. The two datasets diverge only at the edges:
GraSP has Laparoscopic Grasper (absent from EndoVis 2018's type list);
EndoVis 2018 has Ultrasound Probe (absent from GraSP). Ultrasound Probe
instances are skipped -- there is no correct answer for them in our label
space, and scoring them would just be measuring an undefined question, not
a classifier failure.

EndoVis 2018 provides semantic (not instance) segmentation, so instances
are recovered via per-class connected-component analysis on each label
mask. This mirrors GraSP's own Task B crop recipe exactly (bbox-crop, then
mask-multiply to zero out anything in the bbox that isn't this instance,
then optional letterbox pad) so the classifier sees the same kind of input
it was trained on, not a differently-shaped one.

Usage:
    python scripts/evaluate_endovis2018_generalization.py
    python scripts/evaluate_endovis2018_generalization.py --ensemble
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from surgical_ai.data.region_dataset import _pad_to_square
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

# EndoVis 2018 instrument-type category id -> GraSP class name.
# Source: github.com/BCV-Uniandes/ISINet data/robotseg_to_coco.py
# CATEGORIES[:3] + CATEGORIES[5:], re-indexed 1..7, 0=background.
# id 5 (Ultrasound Probe) has no GraSP equivalent and is skipped.
ENDOVIS_ID_TO_GRASP_NAME = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Monopolar Curved Scissors",
    6: "Suction Instrument",
    7: "Clip Applier",
}
MIN_COMPONENT_AREA = 500  # pixels; drops mask fragments/label noise, not real instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endovis-zip", type=Path, default=REPO_ROOT / "EndoVis2018" / "endovis2018.zip")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--ensemble", action="store_true", help="run the full weighted 4-model ensemble instead of a single model")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "experiments" / "region_letterbox_resnet50_320_20260902-234246" / "best.pt")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dump-crops", type=Path, default=None, help="optional dir to save a handful of extracted crops for a sanity check")
    return parser.parse_args()


def extract_instances(zf: zipfile.ZipFile, split: str) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], int, str]]:
    """Returns (image, component_mask, bbox_xywh, grasp_label_idx, source_name) per instance."""
    label_names = sorted(n for n in zf.namelist() if f"/{split}/label/" in n and n.endswith(".bmp"))
    instances = []
    for label_name in label_names:
        image_name = label_name.replace("/label/", "/image/")
        label = np.array(Image.open(io.BytesIO(zf.read(label_name))))
        image = np.array(Image.open(io.BytesIO(zf.read(image_name))).convert("RGB"))
        if label.shape[:2] != image.shape[:2]:
            raise ValueError(f"{label_name} label/image shape mismatch: {label.shape} vs {image.shape}")

        for endovis_id, grasp_name in ENDOVIS_ID_TO_GRASP_NAME.items():
            class_mask = label == endovis_id
            if not class_mask.any():
                continue
            labeled, n_components = ndimage.label(class_mask)
            for comp_id in range(1, n_components + 1):
                comp_mask = labeled == comp_id
                area = comp_mask.sum()
                if area < MIN_COMPONENT_AREA:
                    continue
                ys, xs = np.nonzero(comp_mask)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                instances.append((image, comp_mask, (x0, y0, x1, y1), grasp_name, label_name))
    return instances


def crop_instance(image: np.ndarray, comp_mask: np.ndarray, bbox: tuple[int, int, int, int], letterbox: bool) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = (image[y0:y1, x0:x1] * comp_mask[y0:y1, x0:x1, None]).astype(np.uint8)
    if letterbox:
        crop = _pad_to_square(crop)
    return crop


def run_model(model, images: list[np.ndarray], image_size: int, letterbox: bool, device: torch.device) -> np.ndarray:
    transform = build_transforms(image_size, train=False)
    model.eval()
    all_probs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([transform(Image.fromarray(crop_instance(img, mask, bbox, letterbox))) for img, mask, bbox in batch])
            logits = model(tensors.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}, split: {args.split}")

    zf = zipfile.ZipFile(args.endovis_zip)
    instances = extract_instances(zf, args.split)
    print(f"extracted {len(instances)} instances (>= {MIN_COMPONENT_AREA}px) across the 6 shared classes")

    from collections import Counter
    label_counts = Counter(name for *_x, name, _src in instances)
    for name, count in sorted(label_counts.items()):
        print(f"  {name:<28} {count}")

    if args.dump_crops is not None:
        args.dump_crops.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(instances), size=min(12, len(instances)), replace=False)
        for i, idx in enumerate(sample_idx):
            image, comp_mask, bbox, name, _src = instances[idx]
            crop = crop_instance(image, comp_mask, bbox, letterbox=True)
            safe_name = name.replace(" ", "_")
            Image.fromarray(crop).save(args.dump_crops / f"{i:02d}_{safe_name}.png")
        print(f"dumped {len(sample_idx)} sample crops to {args.dump_crops}")

    class_names = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
                   "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    y_true = np.array([name_to_idx[name] for *_x, name, _src in instances])
    crop_inputs = [(img, mask, bbox) for img, mask, bbox, _name, _src in instances]

    if args.ensemble:
        ensemble_config = yaml.safe_load(args.ensemble_config.read_text())
        members = ensemble_config["members"]
        weight_320 = ensemble_config["weight_resnet50_320"]
        n_rest = len(members) - 1
        w_rest = (1 - weight_320) / n_rest
        weights = [weight_320] + [w_rest] * n_rest
        all_probs = []
        for member in members:
            model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
            model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
            probs = run_model(model, crop_inputs, member["image_size"], member["letterbox"], device)
            all_probs.append(probs)
            standalone_acc = (probs.argmax(axis=1) == y_true).mean()
            print(f"  {member['label']}: standalone accuracy={standalone_acc:.4f}")
        avg = sum(w * p for w, p in zip(weights, all_probs))
        y_pred = avg.argmax(axis=1)
        run_label = f"weighted ensemble (weight_resnet50_320={weight_320})"
    else:
        model = build_model(args.model, num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(REPO_ROOT / args.checkpoint, map_location=device), strict=False)
        probs = run_model(model, crop_inputs, args.image_size, letterbox=True, device=device)
        y_pred = probs.argmax(axis=1)
        run_label = f"single model ({args.model}@{args.image_size}, {args.checkpoint.parent.name})"

    print(f"\n=== {run_label} on EndoVis 2018 ({args.split}), 6 shared classes ===")
    acc = (y_pred == y_true).mean()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(7)), average=None, zero_division=0
    )
    present_classes = [i for i in range(6) if support[i] > 0]
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present_classes)
    print(f"accuracy={acc:.4f} macro-F1({len(present_classes)} classes present in this split)={macro_f1:.4f}")
    for i, name in enumerate(class_names):
        if support[i] == 0:
            note = " (absent from this split)" if i < 6 else " (never appears as ground truth on EndoVis)"
            print(f"  {name:<28} support=0{note}")
            continue
        print(f"  {name:<28} P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f} support={support[i]}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    print("\nconfusion matrix (rows=true, cols=pred), classes=" + str(class_names))
    print(cm)


if __name__ == "__main__":
    main()
