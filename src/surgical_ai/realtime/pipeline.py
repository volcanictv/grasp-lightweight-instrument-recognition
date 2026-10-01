"""Online, causal pipeline for a 1 Hz frame stream.

Per frame, always on: detect, classify each instrument once (single pass per member), score the gate.
Only instruments whose epistemic score reaches the threshold pay for tracking: a background worker follows
the instrument back over the last k buffered frames, classifies those frames and re-labels it. Past frames
are never re-classified unless an instrument is gated, so the always-on cost does not grow with k.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from surgical_ai.realtime.core import BackwardTracker, BufferedFrame, Classifier, Detector
from surgical_ai.realtime.evidence import fuse_frames, label_and_gate


@dataclass
class Refined:
    label: int
    frames_used: int
    seconds: float
    done_at: float


@dataclass
class InstanceResult:
    box: tuple[int, int, int, int]
    label: int
    s1: float
    gated: bool
    refined: Future | None = None


@dataclass
class FrameOutput:
    index: int
    instances: list[InstanceResult]
    detect_ms: float
    classify_ms: float
    done_at: float
    timings: dict[str, float] = field(default_factory=dict)


class RealtimePipeline:
    def __init__(self, detector: Detector, classifier: Classifier, weights: np.ndarray, gate_threshold: float,
                 tracker: BackwardTracker | None = None, look_back: int = 20, synchronous_refine: bool = False,
                 clock=time.monotonic):
        self.detector, self.classifier, self.weights = detector, classifier, np.asarray(weights, dtype=float)
        self.gate_threshold, self.tracker, self.look_back = gate_threshold, tracker, look_back
        self.history: deque[BufferedFrame] = deque(maxlen=look_back)
        # one worker: refinements for a stream are queued in arrival order and a backlog shows up as latency
        self.pool = None if synchronous_refine else ThreadPoolExecutor(max_workers=1)
        self.synchronous_refine = synchronous_refine
        self.clock = clock  # same clock as the stream, so arrival-to-result latency includes queueing

    def process(self, index: int, frame: np.ndarray) -> FrameOutput:
        t0 = time.perf_counter()
        detections = [d for d in self.detector.detect(frame) if d.mask.any()]  # a detector may return empty masks
        t1 = time.perf_counter()
        results, snapshot = [], list(self.history)
        for det in detections:
            logits = self.classifier.logits(frame, det.mask)
            label, s1 = label_and_gate(logits, self.weights)
            gated = self.tracker is not None and s1 >= self.gate_threshold
            result = InstanceResult(det.box, label, s1, gated)
            if gated:
                result.refined = self._submit(snapshot, frame, det.mask, logits)
            results.append(result)
        t2 = time.perf_counter()
        self.history.append(BufferedFrame(index, frame, detections))
        return FrameOutput(index, results, (t1 - t0) * 1000, (t2 - t1) * 1000, self.clock())

    def _submit(self, history: list[BufferedFrame], frame: np.ndarray, mask: np.ndarray, logits: np.ndarray) -> Future:
        if self.synchronous_refine:
            future: Future = Future()
            future.set_result(self._refine(history, frame, mask, logits))
            return future
        return self.pool.submit(self._refine, history, frame, mask, logits)

    def _refine(self, history: list[BufferedFrame], frame: np.ndarray, mask: np.ndarray, logits: np.ndarray) -> Refined:
        start = time.perf_counter()
        past = self.tracker.track_back(history, frame, mask)
        stack = [self.classifier.logits(history[pos].frame, m) for pos, m in sorted(past.items()) if m.any()] + [logits]  # a tracker may report a lost instrument as an empty mask
        label = fuse_frames(np.stack(stack), self.weights)
        return Refined(label, len(stack), time.perf_counter() - start, self.clock())

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True)
