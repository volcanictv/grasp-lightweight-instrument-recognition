"""Concrete detectors, classifier and trackers behind the protocols in core.py. Heavy imports (torch,
ultralytics, sam2) are deferred to construction so the rest of the package imports without them."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from surgical_ai.realtime.core import BufferedFrame, Detection


class YoloDetector:
    """YOLO26-seg as a stand-in for the lab detector; class-agnostic, only masks and scores are used."""

    def __init__(self, weights: Path, device: str = "cuda:0", conf: float = 0.1, imgsz: int = 640):
        from ultralytics import YOLO

        self.model, self.device, self.conf, self.imgsz = YOLO(str(weights)), device, conf, imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, retina_masks=True, device=self.device, verbose=False)[0]
        if result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy().astype(bool)
        scores = result.boxes.conf.cpu().numpy()
        return [Detection(m, float(s)) for m, s in zip(masks, scores) if m.any()]


class StubDetector:
    """Fixed rectangular instruments, for wiring and timing tests with no model and no data."""

    def __init__(self, boxes: Sequence[tuple[int, int, int, int]] = ((10, 10, 90, 60), (120, 40, 200, 110))):
        self.boxes = list(boxes)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        out = []
        for x0, y0, x1, y1 in self.boxes:
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = True
            out.append(Detection(mask))
        return out


def crop_instance(frame: np.ndarray, mask: np.ndarray, letterbox: bool) -> np.ndarray:
    """Mask-multiplied bounding-box crop, padded to a square when the member was trained that way
    (the same recipe as GraspRegionDataset)."""
    from surgical_ai.data.region_dataset import _pad_to_square

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = (frame[y0:y1, x0:x1] * mask[y0:y1, x0:x1, None]).astype(np.uint8)
    return _pad_to_square(crop) if letterbox else crop


class EnsembleClassifier:
    """The evidential members named in `labels`, loaded from an ensemble config; one forward pass each."""

    def __init__(self, ensemble_config: Path, labels: Sequence[str], repo_root: Path, device: str = "cuda:0"):
        import torch
        import yaml

        from surgical_ai.data.transforms import build_transforms
        from surgical_ai.models import build_model

        cfg = yaml.safe_load(Path(ensemble_config).read_text())
        by_label = {m["label"]: m for m in cfg["members"]}
        self.device, self.torch = torch.device(device), torch
        self.members = []
        for label in labels:
            m = by_label[label]
            net = build_model(m["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(self.device)
            net.load_state_dict(torch.load(Path(repo_root) / m["checkpoint"], map_location=self.device), strict=False)
            self.members.append((net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))

    def logits(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        from PIL import Image

        out = []
        with self.torch.no_grad():
            for net, transform, letterbox in self.members:
                image = transform(Image.fromarray(crop_instance(frame, mask, letterbox))).unsqueeze(0).to(self.device)
                out.append(net(image).float().cpu().numpy()[0])
        return np.stack(out)


class StubClassifier:
    """Deterministic logits from the mask size, so the gate fires for some instruments and not others."""

    def __init__(self, members: int = 3, classes: int = 7, seed: int = 0):
        self.members, self.classes, self.seed = members, classes, seed

    def logits(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed + int(mask.sum()))
        out = rng.normal(0, 1, (self.members, self.classes))
        out[:, int(mask.sum()) % self.classes] += 3.0
        return out


class MatchTracker:
    """Follows an instrument back through the detections already made on each buffered frame: IoU matching
    chained frame to frame with a miss budget (`coast`) and a fallback to the current-frame mask. No model runs."""

    def __init__(self, min_iou: float = 0.1, coast: int = 3, center_fallback: bool = True):
        self.min_iou, self.coast, self.center_fallback = min_iou, coast, center_fallback
        self._rle: dict[int, list[dict]] = {}

    def _encode(self, mask: np.ndarray) -> dict:
        from pycocotools import mask as codec

        return codec.encode(np.asfortranarray(mask.astype(np.uint8)))

    def _frame_rles(self, buffered: BufferedFrame) -> list[dict]:
        if buffered.index not in self._rle:
            self._rle[buffered.index] = [self._encode(d.mask) for d in buffered.detections]
            for stale in [i for i in self._rle if i < buffered.index - 1000]:
                del self._rle[stale]
        return self._rle[buffered.index]

    def track_back(self, history: Sequence[BufferedFrame], frame: np.ndarray, mask: np.ndarray) -> dict[int, np.ndarray]:
        from pycocotools import mask as codec

        iou = lambda a, b: float(codec.iou([a], [b], [0])[0][0])
        center = ref = self._encode(mask)
        found: dict[int, np.ndarray] = {}
        misses = 0
        for pos in range(len(history) - 1, -1, -1):
            rles = self._frame_rles(history[pos])
            scored = [(max(iou(ref, r), iou(center, r) if self.center_fallback else 0.0), i) for i, r in enumerate(rles)]
            best = max(scored, default=(0.0, -1))
            if best[1] >= 0 and best[0] >= self.min_iou:
                ref = rles[best[1]]
                found[pos] = history[pos].detections[best[1]].mask
                misses = 0
            else:
                misses += 1
                if misses > self.coast:
                    break
        return found


class SamStyleTracker:
    """SAM2 or EdgeTAM mask propagation (same API; run it from the matching environment). The public API reads
    frames from a directory, so buffered frames are written as JPEGs first and that I/O is part of the measured
    refinement time."""

    def __init__(self, config: str, checkpoint: Path, device: str = "cuda:0"):
        from sam2.build_sam import build_sam2_video_predictor

        self.predictor = build_sam2_video_predictor(config, str(checkpoint), device=device)

    def track_back(self, history: Sequence[BufferedFrame], frame: np.ndarray, mask: np.ndarray) -> dict[int, np.ndarray]:
        from PIL import Image

        tmp = Path(tempfile.mkdtemp(prefix="rt_track_"))
        try:
            for pos, buffered in enumerate(history):
                Image.fromarray(buffered.frame).save(tmp / f"{pos:05d}.jpg", quality=95)
            now = len(history)
            Image.fromarray(frame).save(tmp / f"{now:05d}.jpg", quality=95)
            state = self.predictor.init_state(video_path=str(tmp))
            self.predictor.add_new_mask(state, frame_idx=now, obj_id=1, mask=mask)
            return {idx: (logits[0, 0] > 0).cpu().numpy()
                    for idx, _ids, logits in self.predictor.propagate_in_video(state, reverse=True) if idx < now}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
