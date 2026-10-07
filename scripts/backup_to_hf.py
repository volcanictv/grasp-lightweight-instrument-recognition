"""Backs up the work products that exist only on a working machine to a PRIVATE Hugging Face repository: every classifier checkpoint a config refers to, the segmenter and
tracker weights not published elsewhere, the saved masks, logits and tracks of the GT-box runs, the extracted logits, the stored tracker crops used to train the arm N
classifiers (as tar archives), the experiment logs, and (from the laptop) the local-only docs such as docs/DECISIONS.md. Nothing is made public: the repository is private
because it holds GraSP-derived crops and masks and the internal decision log. Re-running uploads only what changed.

Usage:
    titanxp (repo root):  python scripts/backup_to_hf.py titanxp --repo AryanB005/grasp-pipeline-backup
    laptop (any folder):  python scripts/backup_to_hf.py docs --docs-dir "C:/Users/aryan/Classification Surgurical Tools/docs" --repo AryanB005/grasp-pipeline-backup
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]

README = """# GraSP pipeline backup (private)

Work products behind the paper that live only on the working machines. Private: contains GraSP-derived crops and masks and internal notes.

| path | contents |
|---|---|
| experiments/arm*/, experiments/*_fold*/ ... | every classifier checkpoint (best.pt plus small json/yaml files) that a config in configs/ refers to: dev-fold and official-split members |
| experiments/sam2_gtbox/{tiny_all8,tiny_enc4,small_enc4}/ | SAM2.1-tiny and small segmenter weights (the real-time segmenter is tiny_all8) |
| experiments/yolo26*/weights/ | YOLO26-seg tracker weights (official split: the causal YOLO tracker) |
| experiments/gtbox_sam/ | masks, logits, tracks and summaries of every GT-box run (final, rt_tiny, rt_large, gtft, gtft_ens, causal SAM2 runs) |
| experiments/official_causal/ | tracks and logits of the earlier causal tracker runs |
| experiments_edl/extract/ | extracted single-pass logits used by the analysis scripts |
| archives/ | tar archives of the stored tracker crops (experiments/temporal_neighbors, temporal_neighbors_yolo) used by the arm N training |
| logs/ | experiment logs |
| docs_local/ | the local-only docs of the research repository (DECISIONS.md, findings.md, error analysis, ...) |

Public artefacts are elsewhere: code at github.com/volcanictv/grasp-instrument-classifier and the released weights at AryanB005/grasp-instrument-pipeline.
"""


def tar_dir(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        with tarfile.open(dst, "w") as t:  # PNG crops are already compressed
            t.add(src, arcname=src.name)
    return dst


def referenced_checkpoints() -> list[str]:
    paths = set()
    for pattern in ("configs/arms/*.yaml", "configs/evidential/*.yaml", "configs/*.yaml"):
        for f in REPO_ROOT.glob(pattern):
            for m in re.finditer(r"checkpoint:\s*(\S+)", f.read_text()):
                paths.add(m.group(1))
    return sorted(paths)


def upload(api: HfApi, repo: str, folder: Path, patterns: list[str], dest: str, msg: str) -> None:
    t0 = time.time()
    api.upload_folder(folder_path=str(folder), repo_id=repo, path_in_repo=dest, allow_patterns=patterns, commit_message=msg)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}: {time.time() - t0:.0f}s", flush=True)


def titanxp(api: HfApi, repo: str) -> None:
    ex = REPO_ROOT / "experiments"
    ckpt_dirs = sorted({str(Path(p).parent) for p in referenced_checkpoints()})
    pats = []
    for d in ckpt_dirs:
        rel = Path(d).relative_to("experiments") if d.startswith("experiments") else Path(d)
        pats += [f"{rel}/best.pt", f"{rel}/*.json", f"{rel}/*.yaml", f"{rel}/*.csv"]
    print(len(ckpt_dirs), "checkpoint directories referenced by configs", flush=True)
    upload(api, repo, ex, pats, "experiments", "classifier checkpoints referenced by the configs")
    upload(api, repo, ex, ["sam2_gtbox/tiny_all8/*", "sam2_gtbox/tiny_enc4/*", "sam2_gtbox/small_enc4/*", "sam2_gtbox/*.log"], "experiments", "SAM2 tiny and small segmenters")
    upload(api, repo, ex, ["yolo26*/weights/*.pt", "yolo26*/*.csv", "yolo26*/*.yaml"], "experiments", "YOLO26-seg tracker weights")
    upload(api, repo, ex, ["gtbox_sam/**"], "experiments", "GT-box run artefacts (masks, logits, tracks)")
    upload(api, repo, ex, ["official_causal/**"], "experiments", "earlier causal tracker runs")
    upload(api, repo, REPO_ROOT / "experiments_edl", ["extract/**"], "experiments_edl", "extracted logits")
    upload(api, repo, ex, ["*.log", "*.done", "*.flag"], "logs", "experiment logs")
    tmp = Path.home() / "backup_archives"
    for name in ("temporal_neighbors", "temporal_neighbors_yolo"):
        src = ex / name
        if src.exists():
            tar_dir(src, tmp / f"{name}.tar")
    upload(api, repo, tmp, ["*.tar"], "archives", "stored tracker crops (tar)")


def docs(api: HfApi, repo: str, docs_dir: Path) -> None:
    api.upload_folder(folder_path=str(docs_dir), repo_id=repo, path_in_repo="docs_local", ignore_patterns=["samples/**", "figures/**", "__pycache__/**"],
                      commit_message="local-only docs (DECISIONS.md, findings.md, ...)")
    print("docs uploaded", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["titanxp", "docs"])
    ap.add_argument("--repo", default="AryanB005/grasp-pipeline-backup")
    ap.add_argument("--docs-dir", type=Path, default=None)
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md", repo_id=args.repo, commit_message="inventory")
    titanxp(api, args.repo) if args.mode == "titanxp" else docs(api, args.repo, args.docs_dir)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
