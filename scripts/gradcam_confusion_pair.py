"""Grad-CAM for one confused class pair (2026-09-04 feedback: understand
what the CNN attends to for the top confusion pairs, to inform the next
architecture/data move rather than guessing).

Plain Grad-CAM (Selvaraju et al. 2017), implemented directly with forward/
backward hooks -- no extra dependency. Runs on a single ensemble member
(resnet50_320 by default, the highest-weighted member) rather than the
ensemble as a whole, since Grad-CAM needs one concrete model's gradients;
`docs/error_analysis.md` already found the same confusion pairs recur
across independently trained models, so one model's attention pattern is
informative about the shared cause, not just this one checkpoint's quirk.

For a given (true_class, predicted_class) pair, this model's own
predictions (not the ensemble's) are used to find genuine misclassified
examples of that pair, plus correctly classified examples of the true
class for contrast -- if attention lands somewhere different on the
mistakes than on the correct predictions, that's a real, visible clue
about the failure mode.

Usage:
    python scripts/gradcam_confusion_pair.py \\
        --true-class "Large Needle Driver" --pred-class "Bipolar Forceps" \\
        --out-dir docs/reports/figures/gradcam_needledriver_bipolar
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--true-class", required=True)
    parser.add_argument("--pred-class", required=True)
    parser.add_argument("--ensemble-config", type=Path, default=REPO_ROOT / "configs" / "region_ensemble.yaml")
    parser.add_argument("--member-label", default="resnet50_320", help="which ensemble member (by its 'label' field) to run Grad-CAM on")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-examples", type=int, default=6, help="max misclassified examples of this exact pair to visualize")
    parser.add_argument("--max-correct", type=int, default=4, help="max correctly classified true_class examples to show for contrast")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


class GradCAM:
    """Hooks the last conv block; `__call__` returns a (H, W) heatmap in
    [0, 1] for one image's target-class logit, at that block's own spatial
    resolution (upsampled to the input size by the caller)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, logits: torch.Tensor, target_class: int) -> np.ndarray:
        self_score = logits[0, target_class]
        self_score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def resnet_last_conv_block(model: torch.nn.Module) -> torch.nn.Module:
    # models/classifiers/resnet.py builds a plain torchvision resnet and
    # only replaces .fc -- model.layer4 (the last residual stage) is
    # untouched and keeps its original name.
    return model.layer4


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, out_path: Path) -> None:
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (image.shape[1], image.shape[0]), resample=Image.BILINEAR
    )
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    heat = np.zeros_like(image, dtype=np.float32)
    heat[..., 0] = cam_resized * 255  # red channel carries "attention"
    heat[..., 2] = (1 - cam_resized) * 80  # faint blue in low-attention regions, for contrast

    blended = (0.55 * image.astype(np.float32) + 0.45 * heat).clip(0, 255).astype(np.uint8)
    Image.fromarray(blended).save(out_path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensemble_config = yaml.safe_load(args.ensemble_config.read_text())
    member = next(m for m in ensemble_config["members"] if m["label"] == args.member_label)

    ds = GraspRegionDataset(
        args.data_root, args.split, transform=build_transforms(member["image_size"], train=False),
        letterbox=member["letterbox"],
    )
    class_names = ds.class_names_ordered()
    true_idx = class_names.index(args.true_class)
    pred_idx = class_names.index(args.pred_class)

    model = build_model(member["model"], num_classes=len(class_names), pretrained=False, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(REPO_ROOT / member["checkpoint"], map_location=device), strict=False)
    model.eval()

    target_layer = resnet_last_conv_block(model)
    cam_engine = GradCAM(model, target_layer)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_mistakes, n_correct = 0, 0
    for idx in range(len(ds)):
        file_name, segmentation, box, label_idx = ds.instances[idx]
        if label_idx != true_idx:
            continue

        image_tensor, _label = ds[idx]
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor.unsqueeze(0).to(device))
        pred = int(logits.argmax(dim=1).item())

        want_mistake = pred == pred_idx and n_mistakes < args.max_examples
        want_correct = pred == true_idx and n_correct < args.max_correct
        if not (want_mistake or want_correct):
            continue

        cam = cam_engine(logits, target_class=pred)

        # Undo ImageNet normalization to recover human-viewable crop pixels
        # for the overlay -- same crop the model saw.
        crop_img = np.array(
            Image.fromarray(
                ((image_tensor.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1) * 255).astype(np.uint8)
            )
        )

        tag = "mistake" if want_mistake else "correct"
        count = n_mistakes if want_mistake else n_correct
        out_path = args.out_dir / f"{tag}_{count:02d}_true-{args.true_class.replace(' ', '_')}_pred-{class_names[pred].replace(' ', '_')}.png"
        overlay_heatmap(crop_img, cam, out_path)
        print(f"[{tag}] true={args.true_class} pred={class_names[pred]} -> {out_path.name}")

        if want_mistake:
            n_mistakes += 1
        else:
            n_correct += 1

        if n_mistakes >= args.max_examples and n_correct >= args.max_correct:
            break

    print(f"\nwrote {n_mistakes} mistake overlays + {n_correct} correct overlays to {args.out_dir}")
    if n_mistakes == 0:
        print(f"warning: found zero examples of {args.member_label} itself confusing "
              f"{args.true_class} -> {args.pred_class} -- this pair may be specific to a "
              f"different ensemble member or to the ensemble's combined vote.")


if __name__ == "__main__":
    main()
