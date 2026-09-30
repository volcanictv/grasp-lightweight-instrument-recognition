"""Prep for the 2026-09-29 confirmation runs (docs/DECISIONS.md, "Confirmation runs for the single-pass
evidential default"): which official-test instances need re-tracking so that seeds 43 and 44 (four-member
gate) and the three-member gate can be scored end to end.

Order written to the indices file: first 16 instances that seed 42 already tracked (to check that
re-tracked seed-42 logits reproduce the cached ones), then priority 1 (top-833 by S1 of seeds 43 and 44,
four members, plus seed 42's three-member top-833 members outside its four-member set), then priority 2
(three-member top-833 of seeds 43 and 44 not yet listed).

Usage (on titanxp, repo root):
    python scripts/evidential_seeds_e2e_prep.py --out experiments/evidential_seeds/indices.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

W4 = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}
W3 = {"resnet50_320": 0.40, "resnet50_224": 0.30, "letterbox_crop": 0.30}
BUDGET = 833


def top_set(extract_dir: Path, seed: int, weights: dict) -> list[int]:
    d = np.load(extract_dir / f"E_grasp_official_s{seed}.npz")
    alpha = sum(v * alpha_from_logits(d[f"det_{k}"]) for k, v in weights.items())
    s1 = variance_scores(alpha)["epistemic"]
    return np.argsort(-s1, kind="stable")[:BUDGET].tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, default=REPO_ROOT / "experiments_edl" / "extract")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t4 = {s: top_set(args.extract_dir, s, W4) for s in (42, 43, 44)}
    t3 = {s: top_set(args.extract_dir, s, W3) for s in (42, 43, 44)}
    p1 = list(dict.fromkeys(t4[43] + t4[44] + [i for i in t3[42] if i not in set(t4[42])]))
    in42 = set(t4[42])
    verify = [i for i in p1 if i in in42][:16]
    p2 = [i for i in dict.fromkeys(t3[43] + t3[44]) if i not in set(p1)]
    order = list(dict.fromkeys(verify + p1 + p2))
    print(f"verify {len(verify)}, priority1 {len(p1)}, priority2 extra {len(p2)}, total {len(order)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"indices": [int(i) for i in order], "verify": [int(i) for i in verify],
                                     "priority1": [int(i) for i in p1], "priority2": [int(i) for i in p2],
                                     "top833_four": {str(s): [int(i) for i in v] for s, v in t4.items()},
                                     "top833_three": {str(s): [int(i) for i in v] for s, v in t3.items()}}))


if __name__ == "__main__":
    main()
