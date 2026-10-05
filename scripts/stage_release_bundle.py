"""Stages the weights of the public handoff package under one directory with the file names the package expects, and writes its manifest
(weights_manifest.json: path, size, sha256, role, and a download URL left null until the files are published somewhere).

    evidential_armN_seed{42,43,44}/   the four evidential members of the final configuration (tracker-style-crop training), per seed
    evidential_baseline_seed{...}/    the four members of the earlier configuration (ablation rungs 1 and 2), per seed
    sam2_delta.pt, sam3_delta.pt      the fine-tuned segmenters, trained on all 8 training cases (scripts/make_sam_delta.py)
    sam2_delta_fold2.pt, sam3_delta_fold2.pt   the fold2-trained segmenters of the earlier rungs
    sam2.1_hiera_large.pt             Meta's public SAM2.1-large checkpoint (needed as the base and by the video predictor)

Usage (titanxp, repo root, surgical environment):
    python scripts/stage_release_bundle.py --dest ~/release_bundle --seeds 42 43 44 --sam-deltas ~/release_deltas
    python scripts/stage_release_bundle.py --dest ~/handoff_test --seeds 44 --no-baseline --no-sam      # a small staging for tests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FILE_NAMES = {"resnet50_320": "resnet50_320.pt", "resnet50_224": "resnet50_224.pt", "baseline": "mobilenet_baseline.pt", "letterbox_crop": "mobilenet_letterbox.pt"}
SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True, help="the package root: files go under <dest>/weights/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--no-sam", action="store_true")
    ap.add_argument("--sam-deltas", type=Path, default=None, help="directory with sam2_delta.pt, sam3_delta.pt and the _fold2 ones")
    ap.add_argument("--sam2-base", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    args = ap.parse_args()

    entries = []

    def put(src: Path, rel: str, role: str, url: str | None = None) -> None:
        dst = args.dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        entries.append({"path": rel, "bytes": dst.stat().st_size, "sha256": sha256(dst), "role": role, "url": url})
        print(f"{rel}  {dst.stat().st_size / 1e6:.1f} MB")

    families = [("evidential_armN", "configs/arms/ens4_N_official_s{seed}.yaml", "evidential member, tracker-style-crop training (final configuration)")]
    if not args.no_baseline:
        families.append(("evidential_baseline", "configs/evidential/ens_E_official_s{seed}_lam0p01a10.yaml", "evidential member, baseline training (ablation rungs 1 and 2)"))
    for family, cfg_tpl, role in families:
        for seed in args.seeds:
            cfg = yaml.safe_load((REPO_ROOT / cfg_tpl.format(seed=seed)).read_text())
            for member in cfg["members"]:
                put(REPO_ROOT / member["checkpoint"], f"weights/{family}_seed{seed}/{FILE_NAMES[member['label']]}", f"{role}, seed {seed}, {member['label']}")
    if not args.no_sam:
        for name, role in [("sam2_delta.pt", "fine-tuned SAM2.1-large delta, trained on all 8 training cases"),
                           ("sam3_delta.pt", "fine-tuned SAM3 delta, trained on all 8 training cases (needs the gated facebook/sam3 base)"),
                           ("sam2_delta_fold2.pt", "fine-tuned SAM2.1-large delta of the earlier rungs, fold2 cases only"),
                           ("sam3_delta_fold2.pt", "fine-tuned SAM3 delta of the earlier rungs, fold2 cases only")]:
            if args.sam_deltas and (args.sam_deltas / name).exists():
                put(args.sam_deltas / name, f"weights/{name}", role)
        put(args.sam2_base, "weights/sam2.1_hiera_large.pt", "Meta's public SAM2.1-large checkpoint (base for the delta and for tracking)", SAM2_URL)
    manifest = args.dest / "weights_manifest.json"
    manifest.write_text(json.dumps({"note": "url is null until the files are published; copy them into place by hand until then", "files": entries}, indent=1))
    print(f"wrote {manifest}: {len(entries)} files, {sum(e['bytes'] for e in entries) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
