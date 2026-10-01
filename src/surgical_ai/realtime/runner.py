"""Runs a frame stream through a RealtimePipeline and measures what a live deployment cares about:
per-frame latency against the 1/hz deadline, deadline misses, and how long gated refinements take to land."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from surgical_ai.realtime.pipeline import FrameOutput, RealtimePipeline


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    a = np.array(values)
    return {"n": len(a), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


@dataclass
class StreamReport:
    hz: float
    frames: int
    instances: int
    gated: int
    deadline_misses: int
    frame_latency_ms: dict
    detect_ms: dict
    classify_ms: dict
    refine_latency_ms: dict
    refine_compute_ms: dict
    refined_frames_used: dict
    max_pending_refinements: int
    s1: dict
    instances_per_frame: dict
    s1_values: list
    frame_ms: list
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__


def run_stream(pipeline: RealtimePipeline, arrivals: Iterable[tuple[int, np.ndarray, float]], hz: float,
               max_frames: int | None = None) -> StreamReport:
    """`arrivals` yields (index, frame, arrival_time) on the pipeline's clock (stream.paced or stream.unpaced). Frame latency
    is arrival to the always-on result; refinement latency is arrival to the re-labelled result."""
    deadline = 1.0 / hz
    outputs: list[tuple[FrameOutput, float]] = []
    max_pending, live = 0, []
    for n, (index, frame, arrived) in enumerate(arrivals):
        if max_frames is not None and n >= max_frames:
            break
        out = pipeline.process(index, frame)
        outputs.append((out, arrived))
        live = [f for f in live if not f.done()] + [r.refined for r in out.instances if r.refined is not None]
        max_pending = max(max_pending, len(live))
    pipeline.close()

    frame_latency, refine_latency, refine_compute, used = [], [], [], []
    for out, arrived in outputs:
        frame_latency.append((out.done_at - arrived) * 1000)
        for inst in out.instances:
            if inst.refined is not None:
                r = inst.refined.result()
                refine_compute.append(r.seconds * 1000)
                refine_latency.append((r.done_at - arrived) * 1000)
                used.append(r.frames_used)
    return StreamReport(
        hz=hz, frames=len(outputs), instances=sum(len(o.instances) for o, _ in outputs),
        gated=sum(1 for o, _ in outputs for r in o.instances if r.gated),
        deadline_misses=sum(1 for ms in frame_latency if ms > deadline * 1000),
        frame_latency_ms=percentiles(frame_latency), detect_ms=percentiles([o.detect_ms for o, _ in outputs]),
        classify_ms=percentiles([o.classify_ms for o, _ in outputs]), refine_latency_ms=percentiles(refine_latency),
        refine_compute_ms=percentiles(refine_compute), refined_frames_used=percentiles([float(u) for u in used]),
        max_pending_refinements=max_pending, s1=percentiles([r.s1 for o, _ in outputs for r in o.instances]),
        instances_per_frame=percentiles([float(len(o.instances)) for o, _ in outputs]),
        s1_values=[r.s1 for o, _ in outputs for r in o.instances],
        frame_ms=[[len(o.instances), (o.done_at - a) * 1000, sum(1 for r in o.instances if r.gated)] for o, a in outputs])
