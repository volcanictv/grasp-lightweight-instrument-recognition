import time

import numpy as np
import pytest

from surgical_ai.realtime import stream
from surgical_ai.realtime.adapters import MatchTracker, StubDetector
from surgical_ai.realtime.core import BufferedFrame, Detection
from surgical_ai.realtime.evidence import fuse_frames, label_and_gate
from surgical_ai.realtime.pipeline import RealtimePipeline
from surgical_ai.realtime.runner import run_stream

WEIGHTS = np.array([0.4, 0.3, 0.3])


def frames(n, shape=(60, 80)):
    return [(i, np.zeros((*shape, 3), dtype=np.uint8)) for i in range(n)]


class ByArea:
    """Peaked logits for a large mask (confident), flat for a small one (uncertain)."""

    def logits(self, frame, mask):
        out = np.zeros((3, 7))
        if mask.sum() > 1000:
            out[:, 2] = 8.0
        return out


class SameMaskTracker:
    def track_back(self, history, frame, mask):
        return {pos: mask for pos in range(len(history))}


def test_sample_keeps_every_native_over_hz_th_frame():
    assert [i for i, _ in stream.sample(frames(100), native_fps=30, hz=1)] == [0, 30, 60, 90]
    assert [i for i, _ in stream.sample(frames(5), native_fps=1, hz=1)] == [0, 1, 2, 3, 4]


def test_paced_arrivals_follow_the_schedule_and_never_wait_when_late():
    now = [100.0]
    waits = []

    def sleep(s):
        waits.append(round(s, 3))
        now[0] += s

    arrivals = []
    for n, (_, _, due) in enumerate(stream.paced(frames(4), hz=2, clock=lambda: now[0], sleep=sleep)):
        arrivals.append(due)
        if n == 1:
            now[0] += 5.0  # the consumer stalls for 5 s: the next frames are already late, so no waiting
    assert arrivals == [100.0, 100.5, 101.0, 101.5]
    assert waits == [0.5]


def test_gate_score_is_higher_for_flat_evidence_than_for_agreeing_confident_members():
    _, confident = label_and_gate(np.where(np.eye(7)[2] > 0, 8.0, 0.0)[None].repeat(3, axis=0), WEIGHTS)
    label, flat = label_and_gate(np.zeros((3, 7)), WEIGHTS)
    assert flat > confident
    assert label_and_gate(np.where(np.eye(7)[2] > 0, 8.0, 0.0)[None].repeat(3, axis=0), WEIGHTS)[0] == 2


def test_fuse_frames_lets_confident_frames_outvote_flat_ones():
    sure = np.where(np.eye(7)[4] > 0, 8.0, 0.0)[None].repeat(3, axis=0)
    unsure = np.where(np.eye(7)[1] > 0, 0.6, 0.0)[None].repeat(3, axis=0)
    assert fuse_frames(np.stack([unsure, unsure, sure]), WEIGHTS) == 4


def make_pipeline(threshold, look_back=4, tracker=None):
    detector = StubDetector(boxes=[(0, 0, 40, 40), (50, 0, 60, 10)])  # 1600 px (confident), 100 px (uncertain)
    return RealtimePipeline(detector, ByArea(), WEIGHTS, threshold, tracker or SameMaskTracker(), look_back, synchronous_refine=True)


def test_only_uncertain_instruments_are_refined_and_use_at_most_look_back_frames():
    _, flat = label_and_gate(np.zeros((3, 7)), WEIGHTS)
    pipeline = make_pipeline(threshold=flat / 2, look_back=4)
    outputs = [pipeline.process(i, f) for i, f in frames(8)]
    assert all(len(o.instances) == 2 for o in outputs)
    assert not any(o.instances[0].gated for o in outputs)  # the confident instrument never pays for tracking
    assert all(o.instances[1].gated for o in outputs)
    assert outputs[0].instances[1].refined.result().frames_used == 1  # nothing buffered yet: the current frame only
    assert outputs[7].instances[1].refined.result().frames_used == 5  # 4 buffered past frames plus the current one


def test_empty_detector_masks_are_dropped():
    class Empty:
        def detect(self, frame):
            return [Detection(np.zeros(frame.shape[:2], dtype=bool))]

    out = RealtimePipeline(Empty(), ByArea(), WEIGHTS, 0.0, None, synchronous_refine=True).process(0, frames(1)[0][1])
    assert out.instances == []


def test_no_tracker_means_nothing_is_gated():
    pipeline = RealtimePipeline(StubDetector(), ByArea(), WEIGHTS, 0.0, tracker=None, synchronous_refine=True)
    out = pipeline.process(0, frames(1, shape=(120, 240))[0][1])
    assert not any(r.gated for r in out.instances)


def test_deadline_misses_are_counted_against_one_over_hz():
    class Slow:
        def detect(self, frame):
            time.sleep(0.04)
            return []

    def run(hz):
        pipeline = RealtimePipeline(Slow(), ByArea(), WEIGHTS, 1.0, None, synchronous_refine=True)
        return run_stream(pipeline, stream.unpaced(iter(frames(3))), hz)

    assert run(hz=100).deadline_misses == 3  # 10 ms budget, 40 ms detector
    assert run(hz=2).deadline_misses == 0  # 500 ms budget


def test_refinement_latency_is_reported_for_gated_instances():
    _, flat = label_and_gate(np.zeros((3, 7)), WEIGHTS)
    pipeline = RealtimePipeline(StubDetector(boxes=[(0, 0, 10, 10)]), ByArea(), WEIGHTS, flat / 2, SameMaskTracker(), 3)
    report = run_stream(pipeline, stream.unpaced(iter(frames(5))), hz=1)
    assert report.gated == 5 and report.refine_latency_ms["n"] == 5
    assert report.refine_latency_ms["median"] >= 0


def shifted(dx):
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 10 + dx:30 + dx] = True
    return mask


def test_match_tracker_follows_overlapping_masks_and_stops_at_a_gap():
    pytest.importorskip("pycocotools")
    history = [BufferedFrame(i, np.zeros((40, 60, 3), np.uint8), [Detection(shifted(i))]) for i in range(3)]
    history[0] = BufferedFrame(0, history[0].frame, [])  # the instrument is not detected in the oldest frame
    found = MatchTracker(min_iou=0.3, coast=0).track_back(history, np.zeros((40, 60, 3), np.uint8), shifted(3))
    assert sorted(found) == [1, 2]
    assert (found[2] == shifted(2)).all()
