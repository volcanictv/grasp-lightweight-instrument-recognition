"""Bonus generalization check, EndoVis 2017 (companion to
evaluate_endovis2018_generalization.py -- see that file's docstring for
the shared methodology, not repeated here).

EndoVis 2017's overlap with GraSP is smaller than 2018's: this release
ships its own `instrument_type_mapping.json` (a primary source more
direct than any cross-reference), which maps class 7 to "Other" -- a
catch-all, not "Ultrasound Probe" as a secondary cross-reference
(BCV-Uniandes/ISINet's 2017 category list) would suggest. Only 4 of
GraSP's 7 classes have a same-named counterpart here (Bipolar Forceps,
Prograsp Forceps, Large Needle Driver, Monopolar Curved Scissors) --
no Suction Instrument or Clip Applier at all in this dataset's type task.
Vessel Sealer, Grasping Retractor, and Other are skipped (no GraSP
equivalent).

No single held-out split exists -- the release ships train/ plus 10
cross-validation folds (val1..val10, one held-out sequence each, the
standard 2017 challenge leave-one-out protocol). Evaluated against the
union of all 10 val folds for one aggregate number; not attempting to
reproduce the original per-fold cross-validation protocol since no
training on this data happens here anyway (zero-shot, same as the 2018
check).

Usage:
    python scripts/evaluate_endovis2017_generalization.py
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
from PIL import Image
from scipy import ndimage
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from surgical_ai.data.region_dataset import _pad_to_square
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

# Source: endovis2017/instrument_type_mapping.json, bundled with this
# exact dataset release -- more direct than any secondary cross-reference.
# ids 4 (Vessel Sealer), 5 (Grasping Retractor), 7 (Other) have no GraSP
# equivalent and are skipped.
ENDOVIS_ID_TO_GRASP_NAME = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    6: "Monopolar Curved Scissors",
}
MIN_COMPONENT_AREA = 150  # frames are 512x512, ~0.16x 2018's frame area -- scaled down accordingly


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endovis-zip", type=Path, default=REPO_ROOT / "EndoVis2017" / "endovis2017.zip")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "experiments" / "region_letterbox_resnet50_320_20260902-234246" / "best.pt")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def extract_instances(zf: zipfile.ZipFile) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], str]]:
    all_names = set(zf.namelist())
    val_folders = sorted({n.split("/")[1] for n in all_names if n.startswith("endovis2017/val")})
    label_names = sorted(
        n for n in all_names
        if any(f"/{vf}/label/" in n for vf in val_folders) and n.endswith(".bmp")
    )
    instances = []
    n_missing_image = 0
    for label_name in label_names:
        image_name = label_name.replace("/label/", "/image/")
        if image_name not in all_names:
            n_missing_image += 1
            continue
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
                instances.append((image, comp_mask, (x0, y0, x1, y1), grasp_name))
    if n_missing_image:
        print(f"skipped {n_missing_image}/{len(label_names)} label files with no matching image in the zip "
              "(this release's val folders ship most labels without their image)")
    return instances


def crop_instance(image: np.ndarray, comp_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    crop = (image[y0:y1, x0:x1] * comp_mask[y0:y1, x0:x1, None]).astype(np.uint8)
    return _pad_to_square(crop)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    zf = zipfile.ZipFile(args.endovis_zip)
    instances = extract_instances(zf)
    print(f"extracted {len(instances)} instances (>= {MIN_COMPONENT_AREA}px) across the 4 shared classes")

    from collections import Counter
    label_counts = Counter(name for *_x, name in instances)
    for name, count in sorted(label_counts.items()):
        print(f"  {name:<28} {count}")

    class_names = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
                   "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    y_true = np.array([name_to_idx[name] for *_x, name in instances])

    model = build_model(args.model, num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    transform = build_transforms(args.image_size, train=False)
    all_probs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(instances), batch_size):
            batch = instances[i : i + batch_size]
            tensors = torch.stack([transform(Image.fromarray(crop_instance(img, mask, bbox))) for img, mask, bbox, _name in batch])
            logits = model(tensors.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs)
    y_pred = probs.argmax(axis=1)

    print(f"\n=== single model ({args.model}@{args.image_size}) on EndoVis 2017 (all val folds), 4 shared classes ===")
    acc = (y_pred == y_true).mean()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(7)), average=None, zero_division=0
    )
    present_classes = [i for i in range(4) if support[i] > 0]
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present_classes)
    print(f"accuracy={acc:.4f} macro-F1({len(present_classes)} classes present)={macro_f1:.4f}")
    for i, name in enumerate(class_names):
        if support[i] == 0:
            print(f"  {name:<28} support=0")
            continue
        print(f"  {name:<28} P={precision[i]:.3f} R={recall[i]:.3f} F1={f1[i]:.3f} support={support[i]}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    print("\nconfusion matrix (rows=true, cols=pred), classes=" + str(class_names))
    print(cm)


if __name__ == "__main__":
    main()
