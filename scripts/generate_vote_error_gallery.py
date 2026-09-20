"""Crops and manifest for the error gallery of the vote-based pipeline
(docs/DECISIONS.md 2026-09-20): every instance still wrong after the MC
Dropout-gated tracking, from build_final_pipeline_outputs.py, grouped by
confusion pair. Render with build_error_gallery_page.py.

Display crops use the resnet50_320 member's own preprocessing (letterbox on),
the highest-weighted and highest-resolution member; the other members see
slightly different crops, so this is the most representative single choice.

Usage:
    python scripts/generate_vote_error_gallery.py --out-dir error_gallery_vote
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from surgical_ai.data.region_dataset import GraspRegionDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-instance", type=Path, default=REPO_ROOT / "docs" / "reports" / "final_pipeline" / "per_instance.json")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSp")))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.per_instance.read_text())
    display_ds = GraspRegionDataset(args.data_root, "test", transform=None, letterbox=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in rows:
        pair_dir = args.out_dir / f"{r['true_class'].replace(' ', '_')}_as_{r['pred_class'].replace(' ', '_')}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        image, _label = display_ds[r["idx"]]
        image.thumbnail((256, 256))
        out_path = pair_dir / f"{r['idx']:05d}.png"
        image.save(out_path)
        manifest.append({**r, "image_path": str(out_path.relative_to(args.out_dir)).replace("\\", "/")})
    manifest.sort(key=lambda m: (m["true_class"], m["pred_class"], -m["base_vote_share_pred"]))
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(manifest)} errors; of which broken by tracking: {sum(m['was_correct_before'] for m in manifest)}, "
          f"never tracked: {sum(not m['tracked'] for m in manifest)}")
    for (t, p), n in Counter((m["true_class"], m["pred_class"]) for m in manifest).most_common(8):
        print(f"  {t} -> {p}: {n}")
    print(f"wrote crops + manifest.json to {args.out_dir}")


if __name__ == "__main__":
    main()
