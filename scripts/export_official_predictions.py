"""Per-instance official-test predictions of every configuration in the PI report, so per-class tables,
confusion matrices and the error gallery all come from the same saved numbers.

    base3   three-member evidential alone
    A       base3 + EdgeTAM, causal 20 past frames, gate 2.85e-4
    B       base3 + YOLO26s-seg, causal 15 past frames, gate 5.75e-4
    S       four-member evidential + SAM2-large, 10 past + 10 future frames, gate 1.7e-5 (offline reference)

S is rebuilt from the cached tracked frames with the same code as scripts/evidential_seeds_e2e_eval.py and
checked against the reported 0.9507 / 0.9279.

Usage (titanxp, repo root):
    python scripts/export_official_predictions.py --out docs/reports/causal_realtime/official_predictions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score

import evidential_seeds_e2e_eval as sam_ref
import evidential_three_member_yolo as t
from compare_trackers_causal import load
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores


def causal_predictions(directory: Path, k: int, tau: float, config: Path, extract: Path) -> dict:
    order = t.member_order(config)
    y, base, s1 = t.base_scores(extract)
    tr = load(directory, prefix="official")
    flag = s1 >= tau
    tracked = {}
    for i in np.where(flag)[0]:
        logits, c = tr[int(i)]
        tracked[int(i)] = t.track_predict(logits[max(0, c - k):c + 1], order)
    return {"k": k, "tau": tau, "gated": flag.astype(int).tolist(), "pred": t.apply_gate(base, tracked, flag).tolist()}


def sam2_reference(extract: Path, gate_dir: Path, default_dir: Path, tau: float) -> dict:
    d = np.load(extract)
    y = d["y"]
    weights = sam_ref.CONFIGS["four"]
    alpha = sam_ref.alpha_mix(lambda k: d[f"det_{k}"], weights)
    base = (alpha / alpha.sum(axis=1, keepdims=True)).argmax(axis=1)
    s1 = variance_scores(alpha)["epistemic"]
    frames = sam_ref.load_frames([gate_dir / f"official_frames_shard{s}.npz" for s in (0, 1)]
                                 + [default_dir / f"exp1_frames_shard{s}.npz" for s in (0, 1)], "det_")
    flag = s1 >= tau
    pred = base.copy()
    missing = 0
    for i in np.where(flag)[0]:
        if int(i) in frames:
            pred[i] = sam_ref.track_predict(frames[int(i)], weights)
        else:
            missing += 1
    acc, macro = float((pred == y).mean()), float(f1_score(y, pred, average="macro", labels=range(7)))
    print(f"SAM2 reference rebuilt: accuracy {acc:.4f}, macro-F1 {macro:.4f}, gated {int(flag.sum())}, missing tracks {missing}")
    return {"tau": tau, "gated": flag.astype(int).tolist(), "base": base.tolist(), "pred": pred.tolist(),
            "missing_tracks": missing, "accuracy": acc, "macro_f1": macro}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", type=Path, default=REPO_ROOT / "experiments_edl" / "extract" / "E_grasp_official_s42.npz")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "evidential" / "ens_E_official_s42_lam0p01a10.yaml")
    ap.add_argument("--edgetam-dir", type=Path, default=REPO_ROOT / "experiments" / "official_causal" / "edgetam")
    ap.add_argument("--yolo-dir", type=Path, default=REPO_ROOT / "experiments" / "official_causal" / "yolo26s")
    ap.add_argument("--gate-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_gate_switch")
    ap.add_argument("--default-dir", type=Path, default=REPO_ROOT / "experiments" / "evidential_default")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    y, base3, s1 = t.base_scores(args.extract)
    result = {
        "y": y.tolist(), "s1_3member": s1.tolist(), "base3": base3.tolist(),
        "A": causal_predictions(args.edgetam_dir, 20, 0.000285, args.config, args.extract),
        "B": causal_predictions(args.yolo_dir, 15, 0.000575, args.config, args.extract),
        "S": sam2_reference(args.extract, args.gate_dir, args.default_dir, 1.7e-5),
    }
    for name in ("A", "B"):
        p = np.array(result[name]["pred"])
        print(name, f"accuracy {(p == y).mean():.4f} macro-F1 {f1_score(y, p, average='macro', labels=range(7)):.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
