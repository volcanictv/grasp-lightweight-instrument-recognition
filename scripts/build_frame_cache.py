"""Build a resized on-disk cache of annotated GraSP frames only.

Per PROJECT_SPEC.md sec 5, this is the one authorized exception to "no
duplicated image directories": Milestone 1 loader benchmarking showed the
workstation is loader-bound by >10x versus the model's GPU throughput
(317 img/s loader vs 5258 img/s GPU forward pass, see README.md), so caching
just the ~3449 annotated frames -- not the full 60GB of raw video frames --
at a reduced resolution removes the expensive native 1280x800 decode+resize
from every epoch.

The cache mirrors the source data root's layout (frames-001/frames/<file>,
annotations/ symlinked back to the source) so it's a drop-in --data-root
for train.py and benchmark.py -- no changes to the data pipeline itself.

Usage:
    python scripts/build_frame_cache.py ./GraSP ./GraSP_cache
        [--short-side 256] [--quality 90] [--splits train,test,fold1,fold2]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data import splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing GraSP data root.")
    parser.add_argument("out", type=Path, help="Cache root to create.")
    parser.add_argument("--short-side", type=int, default=256)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--splits", default="train,test,fold1,fold2")
    return parser.parse_args()


def collect_file_names(data_root: Path, split_names: list[str]) -> set[str]:
    names: set[str] = set()
    for split in split_names:
        doc = splits.load_short_term(data_root, split)
        names.update(img["file_name"] for img in doc["images"])
    return names


def resize_one(src: Path, dst: Path, short_side: int, quality: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if min(w, h) > short_side:
            scale = short_side / min(w, h)
            im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        im.save(dst, format="JPEG", quality=quality)
    return dst.stat().st_size


def main() -> None:
    args = parse_args()
    split_names = args.splits.split(",")

    file_names = collect_file_names(args.source, split_names)
    print(f"{len(file_names)} unique annotated frames across splits {split_names}")

    src_frames_root = args.source / "frames-001" / "frames"
    dst_frames_root = args.out / "frames-001" / "frames"

    start = time.time()
    total_bytes = 0
    skipped = 0
    for i, file_name in enumerate(sorted(file_names), 1):
        src = src_frames_root / file_name
        dst = dst_frames_root / file_name
        if dst.exists():
            skipped += 1
            continue
        if not src.exists():
            print(f"warning: missing source frame {src}, skipping")
            continue
        total_bytes += resize_one(src, dst, args.short_side, args.quality)
        if i % 500 == 0:
            print(f"{i}/{len(file_names)} frames cached")
    elapsed = time.time() - start

    annotations_link = args.out / "annotations"
    if not annotations_link.exists():
        annotations_link.symlink_to(
            (args.source / "annotations").resolve(), target_is_directory=True
        )

    print(f"cached {len(file_names) - skipped} frames ({skipped} already present) in {elapsed:.1f}s")
    print(f"new cache bytes written: {total_bytes / (1024 * 1024):.1f} MB")
    print(f"annotations symlinked: {annotations_link} -> {(args.source / 'annotations').resolve()}")
    print(f"cache root: {args.out.resolve()}")


if __name__ == "__main__":
    main()
