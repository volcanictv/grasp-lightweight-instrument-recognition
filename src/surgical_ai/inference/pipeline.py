"""Windowed inference for tracking-by-detection (Milestone 9's second
planned component, docs/DECISIONS.md).

Annotated GraSP frames within a case are ~35 frames apart on average (not
consecutive), too sparse for frame-to-frame tracking on their own -- but
the raw frame directories contain every intermediate frame at native
(~1fps) sampling. This module runs the detector across a small window of
those raw frames around each annotated frame and tracks through it
(`inference/tracking.py`), so a track can survive a few frames of missed
detection (momentary occlusion) and still be present at the annotated
frame we actually have ground truth for.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from surgical_ai.inference.tracking import IOUTracker

_FRAME_NUM_RE = re.compile(r"(\d+)\.jpg$")
_TO_TENSOR = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


def parse_case_and_frame_number(file_name: str) -> tuple[str, int]:
    case = file_name.split("/")[0]
    match = _FRAME_NUM_RE.search(file_name)
    if not match:
        raise ValueError(f"can't parse frame number from '{file_name}'")
    return case, int(match.group(1))


def build_frame_window(
    frames_root: Path, case: str, frame_number: int, radius: int, causal: bool = False
) -> list[tuple[int, Path]]:
    """(frame_number, path) pairs for existing raw frames around
    `frame_number`, sorted ascending. Frames near a case's start/end may
    not exist -- skipped, not padded.

    `causal=False` (default) looks `radius` frames both before and after
    -- this is what a real-time deployment could NOT do (it can't see
    future frames), so it only measures an offline/batch-tracking upper
    bound. `causal=True` looks `radius` frames before only, which is what
    an online system actually has available -- the number to trust for any
    real-time claim.
    """
    start = frame_number - radius
    end = frame_number if causal else frame_number + radius
    window = []
    for n in range(start, end + 1):
        path = frames_root / case / f"{n:05d}.jpg"
        if path.exists():
            window.append((n, path))
    return window


@torch.no_grad()
def run_window_and_track(
    model: torch.nn.Module,
    device: torch.device,
    window: list[tuple[int, Path]],
    target_frame_number: int,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.3,
    max_age: int = 3,
    min_confidence_to_coast: float = 0.5,
    boundary_margin_frac: float = 0.05,
    occlusion_corridor_iou_threshold: float = 0.0,
    occluded_max_age: int = 5,
    require_continuous_occlusion_evidence: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Runs the detector across `window` in temporal order, tracking with
    `IOUTracker`. Returns (direct_detections_at_target, tracked_state_at_target):
    the first is the raw per-frame detector output at the target frame
    alone (the existing, un-tracked baseline); the second is the tracker's
    state at that same frame (includes coasting boxes carried forward from
    a recent real detection).

    The readout happens the instant `n == target_frame_number` is reached
    in the loop, before any later frames in `window` are processed -- so
    even a non-causal (`build_frame_window(..., causal=False)`) window
    never actually influences `tracked_at_target` with future information;
    it only wastes compute processing frames the readout ignores. Verified
    empirically (docs/DECISIONS.md, 2026-09-01): a non-causal and a
    causal-only window produced identical occlusion-recall numbers.
    Callers should pass a causal window and skip the wasted work.
    """
    model.eval()
    tracker: IOUTracker | None = None
    direct_at_target: list[dict] = []
    tracked_at_target: list[dict] = []

    for n, path in window:
        image = Image.open(path).convert("RGB")
        if tracker is None:
            frame_width, frame_height = image.size
            tracker = IOUTracker(
                iou_threshold=iou_threshold, max_age=max_age, min_confidence_to_coast=min_confidence_to_coast,
                frame_width=frame_width, frame_height=frame_height, boundary_margin_frac=boundary_margin_frac,
                occlusion_corridor_iou_threshold=occlusion_corridor_iou_threshold, occluded_max_age=occluded_max_age,
                require_continuous_occlusion_evidence=require_continuous_occlusion_evidence,
            )
        tensor = _TO_TENSOR(image).to(device)
        output = model([tensor])[0]

        detections = []
        for box, label, score in zip(
            output["boxes"].cpu().tolist(), output["labels"].cpu().tolist(), output["scores"].cpu().tolist()
        ):
            if score < score_threshold:
                continue
            detections.append({"box": box, "label": label, "score": score})

        state = tracker.step(detections)
        if n == target_frame_number:
            direct_at_target = detections
            tracked_at_target = state

    return direct_at_target, tracked_at_target
