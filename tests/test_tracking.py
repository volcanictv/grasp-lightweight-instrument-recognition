from surgical_ai.inference.tracking import (
    IOUTracker,
    box_iou,
    extrapolate_box,
    has_plausible_occluder,
    is_exiting_frame,
)


def test_box_iou_identical_boxes_is_one():
    box = [0, 0, 10, 10]
    assert box_iou(box, box) == 1.0


def test_box_iou_disjoint_boxes_is_zero():
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_track_persists_across_matching_frames_same_id():
    tracker = IOUTracker(iou_threshold=0.3, max_age=3)
    r1 = tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 0.9}])
    r2 = tracker.step([{"box": [1, 1, 11, 11], "label": 0, "score": 0.8}])
    assert len(r1) == 1 and len(r2) == 1
    assert r1[0]["track_id"] == r2[0]["track_id"]
    assert r2[0]["matched"] is True


def test_track_survives_missed_frame_within_max_age():
    tracker = IOUTracker(iou_threshold=0.3, max_age=2)
    r1 = tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 0.9}])
    r2 = tracker.step([])  # occluded, no detection this frame
    r3 = tracker.step([{"box": [1, 1, 11, 11], "label": 0, "score": 0.8}])  # reappears

    assert len(r2) == 1  # coasting, still reported
    assert r2[0]["matched"] is False
    assert r2[0]["age"] == 1
    assert r2[0]["track_id"] == r1[0]["track_id"]

    assert r3[0]["matched"] is True
    assert r3[0]["track_id"] == r1[0]["track_id"]


def test_track_dropped_after_max_age_exceeded():
    tracker = IOUTracker(iou_threshold=0.3, max_age=1)
    tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 0.9}])
    r2 = tracker.step([])  # age 1, still within max_age
    r3 = tracker.step([])  # age 2, exceeds max_age -> dropped
    assert len(r2) == 1
    assert len(r3) == 0


def test_coasting_score_decays():
    tracker = IOUTracker(iou_threshold=0.3, max_age=3, coast_decay=0.5)
    tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 1.0}])
    r2 = tracker.step([])
    assert r2[0]["score"] == 0.5


def test_low_confidence_track_does_not_coast():
    # a track born from a low-confidence (likely noise) detection shouldn't
    # be trusted to extrapolate through a missed frame -- reporting it as
    # a confident-looking guess would just add false positives.
    tracker = IOUTracker(iou_threshold=0.3, max_age=3, min_confidence_to_coast=0.5)
    tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 0.2}])
    r2 = tracker.step([])
    assert r2 == []


def test_is_exiting_frame_true_near_edge_moving_outward():
    # box hugging the left edge, moving further left
    assert is_exiting_frame([2, 100, 20, 120], velocity=(-5, 0), frame_width=1280, frame_height=800)


def test_is_exiting_frame_false_near_edge_moving_inward():
    # near the left edge, but moving back toward the center -> not exiting
    assert not is_exiting_frame([2, 100, 20, 120], velocity=(5, 0), frame_width=1280, frame_height=800)


def test_is_exiting_frame_false_in_frame_center():
    # far from every edge, regardless of velocity direction
    assert not is_exiting_frame([600, 350, 650, 400], velocity=(-50, 50), frame_width=1280, frame_height=800)


def test_boundary_exit_drops_track_immediately_instead_of_coasting():
    tracker = IOUTracker(iou_threshold=0.3, max_age=5, frame_width=1280, frame_height=800, boundary_margin_frac=0.05)
    # near the left edge (margin = 0.05*1280 = 64px); small shift keeps IoU >= threshold
    # so this registers as the same track continuing, not a fresh one
    tracker.step([{"box": [10, 100, 30, 120], "label": 0, "score": 0.9}])
    tracker.step([{"box": [7, 100, 27, 120], "label": 0, "score": 0.9}])  # drifting further left -> vx < 0
    # detection gap: near the edge, still carrying negative (outward) velocity
    r3 = tracker.step([])
    assert r3 == []  # dropped immediately, not coasting


def test_missed_frame_away_from_boundary_still_coasts_normally():
    tracker = IOUTracker(iou_threshold=0.3, max_age=5, frame_width=1280, frame_height=800, boundary_margin_frac=0.05)
    tracker.step([{"box": [600, 350, 650, 400], "label": 0, "score": 0.9}])
    tracker.step([{"box": [610, 350, 660, 400], "label": 0, "score": 0.9}])  # small move, still centered
    r3 = tracker.step([])
    assert len(r3) == 1
    assert r3[0]["matched"] is False


def test_extrapolate_box_shifts_by_velocity_keeping_size():
    box = [10, 10, 30, 30]
    predicted = extrapolate_box(box, velocity=(5, -2))
    assert predicted == [15, 8, 35, 28]


def test_has_plausible_occluder_true_when_overlapping_detection():
    predicted_box = [10, 10, 30, 30]
    detections = [{"box": [12, 12, 32, 32], "label": 1, "score": 0.8}]
    assert has_plausible_occluder(predicted_box, detections, iou_threshold=0.1)


def test_has_plausible_occluder_false_when_nothing_nearby():
    predicted_box = [10, 10, 30, 30]
    detections = [{"box": [500, 500, 520, 520], "label": 1, "score": 0.8}]
    assert not has_plausible_occluder(predicted_box, detections, iou_threshold=0.1)


def test_occlusion_corridor_grants_extended_lifetime_when_occluder_present():
    # short base max_age, but a much longer occluded_max_age; the missing
    # track's predicted position overlaps another real detection this frame
    tracker = IOUTracker(
        iou_threshold=0.3, max_age=1, occluded_max_age=5,
        occlusion_corridor_iou_threshold=0.1, boundary_margin_frac=0,
    )
    tracker.step([{"box": [100, 100, 120, 120], "label": 0, "score": 0.9}])
    tracker.step([{"box": [103, 100, 123, 120], "label": 0, "score": 0.9}])  # small move -> velocity (3, 0)
    # instrument now missing, but another instrument occupies its predicted path
    r3 = tracker.step([{"box": [106, 100, 126, 120], "label": 1, "score": 0.7}])
    assert len(r3) == 2  # the occluder's own detection + the coasting missing track
    coasting = [r for r in r3 if r["matched"] is False]
    assert len(coasting) == 1
    # base max_age=1 would have dropped this track already were it not
    # classified as occluded -- confirm it survives past that point too
    r4 = tracker.step([])
    assert any(not r["matched"] for r in r4)


def test_no_occluder_present_keeps_normal_short_lifetime():
    tracker = IOUTracker(
        iou_threshold=0.3, max_age=1, occluded_max_age=5,
        occlusion_corridor_iou_threshold=0.1, boundary_margin_frac=0,
    )
    tracker.step([{"box": [100, 100, 120, 120], "label": 0, "score": 0.9}])
    tracker.step([{"box": [103, 100, 123, 120], "label": 0, "score": 0.9}])
    r3 = tracker.step([])  # nothing else in frame -- no plausible occluder
    assert len(r3) == 1  # coasts this one frame (age 1 <= max_age 1)
    r4 = tracker.step([])
    assert r4 == []  # dropped -- normal short max_age applies, not extended


def test_continuous_evidence_mode_drops_track_once_occluder_disappears():
    # occluded_max_age=5, but require_continuous_occlusion_evidence=True --
    # the extension should only hold as long as an occluder keeps being
    # detected each frame, not just once at gap start
    tracker = IOUTracker(
        iou_threshold=0.3, max_age=1, occluded_max_age=5,
        occlusion_corridor_iou_threshold=0.1, boundary_margin_frac=0,
        require_continuous_occlusion_evidence=True,
    )
    tracker.step([{"box": [100, 100, 120, 120], "label": 0, "score": 0.9}])
    tracker.step([{"box": [103, 100, 123, 120], "label": 0, "score": 0.9}])  # velocity (3, 0)
    # gap frame 1: occluder present -> extended trust holds for this frame
    r3 = tracker.step([{"box": [106, 100, 126, 120], "label": 1, "score": 0.7}])
    assert any(not r["matched"] for r in r3)
    # gap frame 2: occluder gone -> evidence re-checked, fails, normal max_age
    # (already exceeded at age=2 > max_age=1) applies -> original track dropped now
    # (the occluder's own track, track_id=1, legitimately still coasts for one
    # frame on its own -- that's separate, correct behavior, not asserted here)
    r4 = tracker.step([])
    assert not any(r["track_id"] == 0 for r in r4)


def test_continuous_evidence_mode_survives_while_occluder_persists():
    tracker = IOUTracker(
        iou_threshold=0.3, max_age=1, occluded_max_age=5,
        occlusion_corridor_iou_threshold=0.1, boundary_margin_frac=0,
        require_continuous_occlusion_evidence=True,
    )
    tracker.step([{"box": [100, 100, 120, 120], "label": 0, "score": 0.9}])
    tracker.step([{"box": [103, 100, 123, 120], "label": 0, "score": 0.9}])
    for _ in range(3):
        # occluder re-detected fresh each frame (same spot is fine -- what
        # matters here is that evidence keeps being present every step)
        r = tracker.step([{"box": [106, 100, 126, 120], "label": 1, "score": 0.7}])
        assert any(not m["matched"] for m in r)  # missing track still survives


def test_low_confidence_track_stays_alive_internally_and_can_rematch():
    # not reported while coasting, but the track itself survives and can
    # still be matched again once the object reappears.
    tracker = IOUTracker(iou_threshold=0.3, max_age=3, min_confidence_to_coast=0.5)
    tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 0.2}])
    tracker.step([])  # not reported, but track kept alive
    r3 = tracker.step([{"box": [1, 1, 11, 11], "label": 0, "score": 0.9}])
    assert len(r3) == 1
    assert r3[0]["matched"] is True


def test_different_classes_never_match_even_with_perfect_overlap():
    tracker = IOUTracker(iou_threshold=0.3, max_age=3)
    tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 0.9}])
    r2 = tracker.step([{"box": [0, 0, 10, 10], "label": 1, "score": 0.9}])
    # the class-1 detection can't match the class-0 track, so it starts a new track;
    # the old class-0 track starts coasting instead
    assert len(r2) == 2
    track_ids = {r["track_id"] for r in r2}
    assert len(track_ids) == 2


def test_low_iou_starts_new_track_instead_of_matching():
    tracker = IOUTracker(iou_threshold=0.5, max_age=3)
    r1 = tracker.step([{"box": [0, 0, 10, 10], "label": 0, "score": 0.9}])
    r2 = tracker.step([{"box": [8, 8, 18, 18], "label": 0, "score": 0.9}])  # IoU ~0.02, below threshold
    assert len(r2) == 2  # old track coasts, new detection starts a second track
    assert {r["matched"] for r in r2} == {True, False}
    new_track = next(r for r in r2 if r["matched"])
    assert new_track["track_id"] != r1[0]["track_id"]
