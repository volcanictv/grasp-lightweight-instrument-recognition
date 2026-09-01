"""Lightweight tracking-by-detection (Milestone 9's second planned
component, docs/DECISIONS.md).

Base algorithm is the IOU-Tracker (Bochinski et al. 2017, "High-Speed
Tracking-by-Detection Without Using Image Information"): greedy box-IoU
association between consecutive frames, no learned motion model, no
appearance features -- deliberately the simplest tracker that could
plausibly help, consistent with this project's efficiency thesis
(near-zero added compute on top of an existing detector). Two additions
on top of the original paper, both needed once a detection gap is
supposed to survive *momentary* occlusion instead of ending the track
immediately:

  - `max_age`: a track survives up to this many consecutive unmatched
    frames before being dropped (the same "max age" idea SORT/DeepSORT use).
  - `min_confidence_to_coast` and boundary-exit detection (below): both
    exist because letting *any* track coast through a gap, unconditionally,
    measurably hurt mAP@50 in practice (docs/DECISIONS.md, 2026-09-01) --
    a coasted box is a stale guess, and it is only a good guess if the gap
    is really occlusion, not the instrument leaving the frame or being a
    low-confidence (likely noise) track to begin with. Both are
    precision/recall tradeoffs to tune, not free improvements.

`min_confidence_to_coast`: a track can only be *reported* while coasting
if its last real detection scored at least this. A track born from a
single low-confidence detection is unreliable to extrapolate through a
gap; it still survives internally (can be re-matched later) but is not
trusted to produce an output while unconfirmed.

Boundary-exit detection: tracks a coarse per-frame velocity (box-center
delta between the last two real matches). When a track goes unmatched, if
its last box was near a frame edge *and* its velocity was carrying it
further toward that edge, the gap is treated as the instrument genuinely
leaving the field of view rather than momentary occlusion -- the track is
dropped immediately instead of coasting. This targets the other failure
mode found alongside low-confidence coasting: a real, well-tracked
instrument that actually exits the frame still produces a confident-
looking but wrong coasted box if nothing distinguishes "gone" from
"occluded". Lightweight, no learned parameters, no detector changes --
pure post-processing.

Occlusion-corridor detection: the positive counterpart to boundary-exit's
negative signal. On the first missed frame of a gap, the track's box is
extrapolated one step forward by its velocity; if that predicted position
overlaps another instrument actually detected this frame, the gap looks
like plausible occlusion by that instrument (not disappearance), and the
track is granted a longer `occluded_max_age` instead of the normal, short
`max_age` for the rest of the gap. A track with no such corroborating
evidence gets the normal (short) lifetime -- this is deliberately
asymmetric: extended trust requires positive geometric evidence of an
occluder, not just the absence of a boundary-exit signal.
"""

from __future__ import annotations

from dataclasses import dataclass


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_exiting_frame(
    box: list[float], velocity: tuple[float, float], frame_width: float, frame_height: float, margin_frac: float = 0.05
) -> bool:
    """True if `box` sits within `margin_frac` of a frame edge and
    `velocity` (box-center delta per frame) points further past that same
    edge -- i.e. the object looks like it's leaving the frame, not just
    momentarily hidden. Checked independently per edge (an object can only
    be "exiting" through one edge at a time in practice, but nothing stops
    checking all four).
    """
    margin_x, margin_y = frame_width * margin_frac, frame_height * margin_frac
    x1, y1, x2, y2 = box
    vx, vy = velocity
    exiting_left = x1 <= margin_x and vx < 0
    exiting_right = x2 >= frame_width - margin_x and vx > 0
    exiting_top = y1 <= margin_y and vy < 0
    exiting_bottom = y2 >= frame_height - margin_y and vy > 0
    return exiting_left or exiting_right or exiting_top or exiting_bottom


def extrapolate_box(box: list[float], velocity: tuple[float, float]) -> list[float]:
    """One-step linear extrapolation: shift `box` by `velocity` (a
    box-center delta), keeping its size fixed. Deliberately the simplest
    possible motion model -- no acceleration, no learned parameters.
    """
    x1, y1, x2, y2 = box
    vx, vy = velocity
    return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]


def has_plausible_occluder(predicted_box: list[float], detections: list[dict], iou_threshold: float = 0.1) -> bool:
    """True if `predicted_box` (a missing track's extrapolated position)
    overlaps any of this frame's actual detections at least `iou_threshold`
    -- i.e. something is really there that could be occluding the missing
    instrument, not just empty background. Checked against every detection
    in the frame regardless of class: an occluder is whatever object is
    physically in the way, not necessarily the same instrument type.
    """
    return any(box_iou(predicted_box, d["box"]) >= iou_threshold for d in detections)


@dataclass
class Track:
    track_id: int
    label: int
    box: list[float]
    score: float
    age: int = 0  # consecutive frames since last matched to a real detection
    velocity: tuple[float, float] = (0.0, 0.0)  # box-center delta, per frame, from the last real match
    likely_occluded: bool = False  # set once per gap, from the occlusion-corridor check at the first missed frame


def _box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


class IOUTracker:
    """Call `step` once per frame, in temporal order, with that frame's
    post-NMS detections (`{"box": [x1,y1,x2,y2], "label": int, "score": float}`).
    Returns the current state of every active, reportable track (matched
    this frame or still coasting), one dict per track, same keys plus
    `track_id`, `matched` (bool), and `age` (frames since last real match).

    `frame_width`/`frame_height` are only used for boundary-exit detection;
    pass `boundary_margin_frac=0` (or leave frame size at its defaults on a
    dataset where frame size doesn't matter) to disable that check.

    `occlusion_corridor_iou_threshold=0` disables occlusion-corridor
    detection (every gap gets the normal, short `max_age`). Set it
    positive to let a track earn `occluded_max_age` instead, but only for
    a gap that starts with a plausible occluder actually present.

    `require_continuous_occlusion_evidence`: if False (default), the
    corridor check runs once at the start of a gap and the decision
    (`likely_occluded`) sticks for the rest of it -- but that evidence can
    go stale exactly like the coasted box itself does, since it's no
    guarantee the "occluder" or the missing instrument stay put for
    several more frames. If True, the check re-runs on *every* missed
    frame using that frame's own detections, so extended trust requires
    evidence to keep holding up, not just have held once at the start.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 3,
        coast_decay: float = 0.9,
        min_confidence_to_coast: float = 0.5,
        frame_width: float = 1280.0,
        frame_height: float = 800.0,
        boundary_margin_frac: float = 0.05,
        occlusion_corridor_iou_threshold: float = 0.0,
        occluded_max_age: int = 5,
        require_continuous_occlusion_evidence: bool = False,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.coast_decay = coast_decay
        self.min_confidence_to_coast = min_confidence_to_coast
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.boundary_margin_frac = boundary_margin_frac
        self.occlusion_corridor_iou_threshold = occlusion_corridor_iou_threshold
        self.occluded_max_age = occluded_max_age
        self.require_continuous_occlusion_evidence = require_continuous_occlusion_evidence
        self.tracks: list[Track] = []
        self._next_id = 0

    def step(self, detections: list[dict]) -> list[dict]:
        unmatched = set(range(len(detections)))
        results: list[dict] = []
        survivors: list[Track] = []

        for track in self.tracks:
            best_iou, best_j = 0.0, -1
            for j in unmatched:
                d = detections[j]
                if d["label"] != track.label:
                    continue
                iou = box_iou(track.box, d["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= self.iou_threshold:
                d = detections[best_j]
                track.velocity = tuple(
                    new - old for new, old in zip(_box_center(d["box"]), _box_center(track.box))
                )
                track.box, track.score, track.age = d["box"], d["score"], 0
                track.likely_occluded = False
                unmatched.discard(best_j)
                survivors.append(track)
                results.append(
                    {"box": track.box, "label": track.label, "score": track.score,
                     "track_id": track.track_id, "matched": True, "age": 0}
                )
                continue

            if self.boundary_margin_frac > 0 and is_exiting_frame(
                track.box, track.velocity, self.frame_width, self.frame_height, self.boundary_margin_frac
            ):
                continue  # treated as a real exit, not occlusion -- drop the track now

            check_corridor_now = self.occlusion_corridor_iou_threshold > 0 and (
                track.age == 0 or self.require_continuous_occlusion_evidence
            )
            if check_corridor_now:
                # either the first missed frame of a new gap, or (with
                # require_continuous_occlusion_evidence) every missed frame --
                # decide (or re-decide) whether this frame looks like
                # occlusion, using its real detections as evidence
                predicted_box = extrapolate_box(track.box, track.velocity)
                track.likely_occluded = has_plausible_occluder(
                    predicted_box, detections, self.occlusion_corridor_iou_threshold
                )

            track.age += 1
            effective_max_age = self.occluded_max_age if track.likely_occluded else self.max_age
            if track.age <= effective_max_age:
                survivors.append(track)
                if track.score >= self.min_confidence_to_coast:
                    results.append(
                        {"box": track.box, "label": track.label, "score": track.score * self.coast_decay,
                         "track_id": track.track_id, "matched": False, "age": track.age}
                    )
                track.score *= self.coast_decay

        self.tracks = survivors

        for j in unmatched:
            d = detections[j]
            track = Track(track_id=self._next_id, label=d["label"], box=d["box"], score=d["score"])
            self._next_id += 1
            self.tracks.append(track)
            results.append(
                {"box": d["box"], "label": d["label"], "score": d["score"],
                 "track_id": track.track_id, "matched": True, "age": 0}
            )
        return results
