from pathlib import Path

from surgical_ai.inference.pipeline import build_frame_window, parse_case_and_frame_number


def test_parse_case_and_frame_number():
    case, n = parse_case_and_frame_number("CASE001/00036.jpg")
    assert case == "CASE001"
    assert n == 36


def test_build_frame_window_skips_missing_frames(tmp_path: Path):
    case_dir = tmp_path / "CASE001"
    case_dir.mkdir()
    for n in [33, 34, 36, 37]:  # 35 (the center) and its neighbors partially missing
        (case_dir / f"{n:05d}.jpg").touch()

    window = build_frame_window(tmp_path, "CASE001", frame_number=35, radius=2)
    numbers = [n for n, _ in window]
    assert numbers == [33, 34, 36, 37]  # 35 itself doesn't exist, correctly skipped


def test_build_frame_window_sorted_ascending(tmp_path: Path):
    case_dir = tmp_path / "CASE001"
    case_dir.mkdir()
    for n in [10, 11, 12, 13, 14]:
        (case_dir / f"{n:05d}.jpg").touch()

    window = build_frame_window(tmp_path, "CASE001", frame_number=12, radius=2)
    numbers = [n for n, _ in window]
    assert numbers == [10, 11, 12, 13, 14]


def test_build_frame_window_causal_excludes_future_frames(tmp_path: Path):
    case_dir = tmp_path / "CASE001"
    case_dir.mkdir()
    for n in [10, 11, 12, 13, 14]:
        (case_dir / f"{n:05d}.jpg").touch()

    window = build_frame_window(tmp_path, "CASE001", frame_number=12, radius=2, causal=True)
    numbers = [n for n, _ in window]
    assert numbers == [10, 11, 12]  # nothing after frame 12
