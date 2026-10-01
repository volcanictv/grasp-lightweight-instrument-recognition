"""Types and the two plug-in points of the real-time pipeline.

A `Detector` is whatever finds instruments in a frame (the lab detector, or the YOLO26-seg stand-in);
a `BackwardTracker` follows one instrument over buffered past frames. Everything else in this package
only talks to these two protocols, so swapping either one touches nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Detection:
    mask: np.ndarray  # bool, H x W
    score: float = 1.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(self.mask)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass
class BufferedFrame:
    index: int
    frame: np.ndarray  # H x W x 3, RGB uint8
    detections: list[Detection] = field(default_factory=list)


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class Classifier(Protocol):
    def logits(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Raw logits of every ensemble member for the instrument under `mask`: (members, classes)."""
        ...


class BackwardTracker(Protocol):
    def track_back(self, history: Sequence[BufferedFrame], frame: np.ndarray, mask: np.ndarray) -> dict[int, np.ndarray]:
        """Follow the instrument at `mask` in the current `frame` back through `history` (oldest first,
        current frame excluded). Returns {position in history: bool mask}; frames where it is lost are absent."""
        ...
