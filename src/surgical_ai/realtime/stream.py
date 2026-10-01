"""Frame sources and 1 Hz sampling of a higher-rate video, with optional wall-clock pacing so a test can
behave like a live feed (frames arrive on a schedule and a slow pipeline falls behind)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np


def sample(frames: Iterable[tuple[int, np.ndarray]], native_fps: float, hz: float) -> Iterator[tuple[int, np.ndarray]]:
    """Keep every (native_fps / hz)-th frame. native_fps 30 and hz 1 keeps frame 0, 30, 60, ..."""
    step = max(1, round(native_fps / hz))
    for index, frame in frames:
        if index % step == 0:
            yield index, frame


def paced(frames: Iterable[tuple[int, np.ndarray]], hz: float, clock: Callable[[], float] = time.monotonic,
          sleep: Callable[[float], None] = time.sleep) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield (index, frame, arrival_time) with arrivals 1/hz apart on `clock`, sleeping until each is due.
    A consumer that takes longer than 1/hz receives the next frame late, which is what a live feed does."""
    start = clock()
    for n, (index, frame) in enumerate(frames):
        due = start + n / hz
        wait = due - clock()
        if wait > 0:
            sleep(wait)
        yield index, frame, due


def unpaced(frames: Iterable[tuple[int, np.ndarray]], clock: Callable[[], float] = time.monotonic) -> Iterator[tuple[int, np.ndarray, float]]:
    """Compute-only mode: no waiting, arrival is whenever the pipeline asks for the frame."""
    for index, frame in frames:
        yield index, frame, clock()


def directory_source(directory: Path) -> Iterator[tuple[int, np.ndarray]]:
    from PIL import Image

    for index, path in enumerate(sorted(Path(directory).glob("*.jpg"))):
        yield index, np.array(Image.open(path).convert("RGB"))


def video_source(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()


def synthetic_source(n_frames: int, size: tuple[int, int] = (240, 320), seed: int = 0) -> Iterator[tuple[int, np.ndarray]]:
    """Noise frames, enough to exercise timing and wiring without any data on disk."""
    rng = np.random.default_rng(seed)
    for index in range(n_frames):
        yield index, rng.integers(0, 255, (*size, 3), dtype=np.uint8)
