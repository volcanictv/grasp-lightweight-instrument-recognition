"""Challenge-stratified instance classification: TAPIS with ground-truth boxes against our GT-box pipeline on the same 2,861 official-test instruments.

Definitions of the strata are in docs/reports/challenge_strata/protocol.md and were fixed before any accuracy was computed. Inputs:
  --feat-test / --feat-train   scripts/challenge_strata_features.py outputs
  --transitions                refinement_transitions.py csv of the top-941 run (per seed: single pass, refine-all, and the selected set)
  --tapis                      TAPIS epoch_0_preds_instruments.json from the GT-box evaluation
  --cv-logits                  folder with logits_f*.npz of the 5-fold train-case CV (similar-class pairs)
Writes results.json, strata_table.md and per_instrument.csv into --out-dir.

    python scripts/challenge_strata_eval.py --feat-test ... --feat-train ... --transitions ... --tapis ... --cv-logits ... --out-dir docs/reports/challenge_strata
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
SEEDS = ["42", "43", "44"]
PAIR_RATE = 0.03
OCC_BOX = 0.10
B = 10000


def load(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: (np.array([r[k] for r in rows]) if k in ("file", "case") else np.array([float(r[k]) for r in rows])) for k in rows[0]}


def case_top(case: np.ndarray, score: np.ndarray, q: float = 0.9) -> np.ndarray:
    out = np.zeros(len(score), bool)
    for c in np.unique(case):
        m = case == c
        out[m] = score[m] >= np.quantile(score[m], q)
    return out


def confusable_pairs(cv_dir: Path) -> tuple[list[tuple[int, int]], np.ndarray]:
    conf = np.zeros((7, 7), int)
    for f in sorted(glob.glob(str(cv_dir / "logits_f*.npz"))):
        z = np.load(f)
        p = 0
        for k in (k for k in z.files if k.startswith("det_")):
            e = np.exp(z[k] - z[k].max(1, keepdims=True))
            p = p + e / e.sum(1, keepdims=True)
        for a, b in zip(z["y"], p.argmax(1)):
            conf[a, b] += 1
    rate = conf / conf.sum(1, keepdims=True)
    pairs = [(a, b) for a in range(7) for b in range(a + 1, 7) if max(rate[a, b], rate[b, a]) >= PAIR_RATE]
    return pairs, conf


def macro_f1_weighted(W: np.ndarray, y: np.ndarray, pred: np.ndarray, mask: np.ndarray, classes: list[int]) -> np.ndarray:
    f1 = []
    for c in classes:
        tp = W @ (mask & (y == c) & (pred == c))
        fp = W @ (mask & (y != c) & (pred == c))
        fn = W @ (mask & (y == c) & (pred != c))
        f1.append(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0))
    return np.mean(f1, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat-test", type=Path, required=True)
    ap.add_argument("--feat-train", type=Path, required=True)
    ap.add_argument("--transitions", type=Path, required=True)
    ap.add_argument("--tapis", type=Path, required=True)
    ap.add_argument("--cv-logits", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=B)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    te, tr = load(args.feat_test), load(args.feat_train)
    n = len(te["case"])
    y = te["label"].astype(int)

    thr = {"specular_frac_p90_train": float(np.quantile(tr["specular_frac"], 0.9)), "edge_frac_p90_train": float(np.quantile(tr["edge_frac"], 0.9))}
    small_thr = {int(c): float(np.quantile(tr["area_frac"][tr["label"] == c], 0.1)) for c in range(7)}
    haze_score = np.zeros(n)
    for c in np.unique(te["case"]):
        m = te["case"] == c
        haze_score[m] = (rankdata(-te["ring_sat"][m]) + rankdata(-te["ring_contrast"][m])) / (2 * m.sum())
    pairs, cv_conf = confusable_pairs(args.cv_logits)

    flags = {
        "occ_instrument": te["other_in_box"] >= OCC_BOX,
        "occ_fragmented": te["fragmented"] > 0,
        "small_visible": np.array([te["area_frac"][i] <= small_thr[int(y[i])] for i in range(n)]),
        "blood": case_top(te["case"], te["blood_frac"]),
        "specular": te["specular_frac"] >= thr["specular_frac_p90_train"],
        "lowcontrast_haze": case_top(te["case"], haze_score),
        "corner": te["n_borders"] >= 2,
        "edge_dominant": te["edge_frac"] >= thr["edge_frac_p90_train"],
    }
    occ_any = flags["occ_instrument"] | flags["occ_fragmented"] | flags["small_visible"]
    deg_any = flags["blood"] | flags["specular"] | flags["lowcontrast_haze"]
    bnd_any = flags["corner"] | flags["edge_dominant"]
    flags.update({"occlusion_any": occ_any, "degraded_any": deg_any, "boundary_any": bnd_any})
    flags["clean"] = ~(occ_any | deg_any | bnd_any)
    flags["compound"] = (occ_any.astype(int) + deg_any + bnd_any) >= 2

    pair_in = np.zeros(n, bool)
    pair_rows = {}
    for a, b in pairs:
        m = (y == a) | (y == b)
        pair_rows[f"pair: {CLASSES[a]} / {CLASSES[b]}"] = m
        pair_in |= m
    clear = ~pair_in

    strata: dict[str, tuple[str, np.ndarray]] = {"All instruments": ("overall", np.ones(n, bool))}
    groups = {"occlusion": ["occ_instrument", "occ_fragmented", "small_visible", "occlusion_any"],
              "degraded view": ["blood", "specular", "lowcontrast_haze", "degraded_any"],
              "frame boundary": ["corner", "edge_dominant", "boundary_any"]}
    for g, names in groups.items():
        for nm in names:
            strata[nm] = (g, flags[nm])
    for nm, m in pair_rows.items():
        strata[nm] = ("similar classes", m)
    strata["any class in a confusable pair"] = ("similar classes", pair_in)
    strata["clear classes (no pair >= 3%)"] = ("similar classes", clear)
    hard = occ_any | flags["blood"] | flags["lowcontrast_haze"] | bnd_any
    strata["hard: occlusion, blood, haze, boundary"] = ("combined", hard)
    strata["hard, without glare-flagged instruments"] = ("combined", hard & ~flags["specular"])
    strata["compound (two or more groups)"] = ("combined", flags["compound"])
    strata["clean (no flag)"] = ("combined", flags["clean"])
    for c in range(7):
        strata[f"class: {CLASSES[c]}"] = ("per class", y == c)

    rows = list(csv.DictReader(open(args.transitions)))
    key = {(te["case"][i], int(te["box_x"][i]), int(te["box_y"][i]), int(te["box_w"][i]), int(te["box_h"][i])): i for i in range(n)}
    assert len(key) == n
    ours = {"single": {}, "final": {}, "refine_all": {}}
    for s in SEEDS:
        rs = [r for r in rows if r["seed"] == s]
        assert len(rs) == n
        for nm in ours:
            ours[nm][s] = np.full(n, -1)
        for r in rs:
            i = int(r["instrument"])
            x, yy, w, h = (int(v) for v in r["box_xywh"].split())
            assert key[(r["case"], x, yy, w, h)] == i and int(r["gt"]) == y[i], "instrument identity mismatch"
            ours["single"][s][i] = int(r["single_pred"])
            ours["refine_all"][s][i] = int(r["refine_all_pred"])
            ours["final"][s][i] = int(r["refine_all_pred"]) if r["sel_s1_top833"] == "True" else int(r["single_pred"])
        assert (ours["final"][s] >= 0).all() and sum(r["sel_s1_top833"] == "True" for r in rs) == 941

    tp = json.load(open(args.tapis))
    tapis = np.full(n, -1)
    for f, v in tp.items():
        case = f.split("/")[0]
        for ins in v["instances"]:
            x1, y1, x2, y2 = ins["bbox"]
            tapis[key[(case, int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))]] = int(np.argmax(ins["instruments_score_dist"]))
    assert (tapis >= 0).all()

    # frame-cluster bootstrap weights, shared by every system and stratum
    frame_id = {}
    fidx = np.array([frame_id.setdefault((te["case"][i], te["file"][i]), len(frame_id)) for i in range(n)])
    rng = np.random.default_rng(0)
    nf = len(frame_id)
    cnt = rng.multinomial(nf, np.full(nf, 1 / nf), size=args.boot).astype(np.float32)
    W = cnt[:, fidx]

    systems = {"tapis": [tapis], "ours_single": [ours["single"][s] for s in SEEDS], "ours_final": [ours["final"][s] for s in SEEDS], "ours_refine_all": [ours["refine_all"][s] for s in SEEDS]}

    def metrics(preds: list[np.ndarray], mask: np.ndarray, classes: list[int]) -> tuple[float, float, np.ndarray, np.ndarray, list[float], list[float]]:
        acc_b = np.mean([(W @ (mask & (p == y))) / np.maximum(W @ mask, 1e-9) for p in preds], axis=0)
        f1_b = np.mean([macro_f1_weighted(W, y, p, mask, classes) for p in preds], axis=0)
        acc_s = [float(((p == y) & mask).sum() / mask.sum()) for p in preds]
        f1_s = []
        one = np.ones((1, n), np.float32)
        for p in preds:
            f1_s.append(float(macro_f1_weighted(one, y, p, mask, classes)[0]))
        return float(np.mean(acc_s)), float(np.mean(f1_s)), acc_b, f1_b, acc_s, f1_s

    def ci(a: np.ndarray) -> list[float]:
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]

    results = {"protocol": "docs/reports/challenge_strata/protocol.md", "n_instruments": n, "n_frames": nf, "bootstrap_resamples": args.boot,
               "thresholds": {**thr, "small_visible_area_frac_p10_train_by_class": {CLASSES[c]: small_thr[c] for c in range(7)}, "occ_instrument_box_cover": OCC_BOX,
                              "confusable_pairs": [[CLASSES[a], CLASSES[b]] for a, b in pairs], "pair_rate_threshold": PAIR_RATE},
               "cv_confusion_train_cases": cv_conf.tolist(), "strata": []}
    for name, (group, mask) in strata.items():
        mask = mask.astype(bool)
        cls = [c for c in range(7) if (mask & (y == c)).any()]
        entry = {"stratum": name, "group": group, "n_instruments": int(mask.sum()), "n_frames": int(len(set(fidx[mask]))), "n_cases": int(len(set(te["case"][mask]))),
                 "class_counts": {CLASSES[c]: int((mask & (y == c)).sum()) for c in cls}, "systems": {}, "diffs": {}}
        boot = {}
        for sysname, preds in systems.items():
            a, f, ab, fb, a_seeds, f_seeds = metrics(preds, mask, cls)
            boot[sysname] = (ab, fb)
            entry["systems"][sysname] = {"acc": a, "acc_ci": ci(ab), "f1": f, "f1_ci": ci(fb), "acc_seed_range": [min(a_seeds), max(a_seeds)], "f1_seed_range": [min(f_seeds), max(f_seeds)]}
        for a_name, b_name in [("ours_final", "tapis"), ("ours_final", "ours_single"), ("ours_single", "tapis")]:
            d_acc, d_f1 = boot[a_name][0] - boot[b_name][0], boot[a_name][1] - boot[b_name][1]
            entry["diffs"][f"{a_name}-{b_name}"] = {"acc": entry["systems"][a_name]["acc"] - entry["systems"][b_name]["acc"], "acc_ci": ci(d_acc),
                                                    "f1": entry["systems"][a_name]["f1"] - entry["systems"][b_name]["f1"], "f1_ci": ci(d_f1)}
        t_ok = tapis == y
        both = []
        for p in systems["ours_final"]:
            o_ok = p == y
            both.append([int((mask & o_ok & ~t_ok).sum()), int((mask & ~o_ok & t_ok).sum()), int((mask & o_ok & t_ok).sum()), int((mask & ~o_ok & ~t_ok).sum())])
        entry["discordant_ours_final_vs_tapis_mean_over_seeds"] = {"ours_right_tapis_wrong": float(np.mean([b[0] for b in both])), "ours_wrong_tapis_right": float(np.mean([b[1] for b in both])),
                                                                   "both_right": float(np.mean([b[2] for b in both])), "both_wrong": float(np.mean([b[3] for b in both]))}
        results["strata"].append(entry)

    (args.out_dir / "results.json").write_text(json.dumps(results, indent=1))

    def pc(v: float) -> str:
        return f"{100 * v:.1f}"

    def cell(s: dict, key: str) -> str:
        return f"{pc(s[key])} [{pc(s[key + '_ci'][0])}, {pc(s[key + '_ci'][1])}]"

    def dcell(d: dict, key: str) -> str:
        return f"{100 * d[key]:+.1f} [{100 * d[key + '_ci'][0]:+.1f}, {100 * d[key + '_ci'][1]:+.1f}]"

    lines = ["| Stratum | n inst. | n frames | TAPIS acc | Ours single acc | Ours final acc | Final - TAPIS acc | Final - single acc | TAPIS F1 | Ours final F1 | Final - TAPIS F1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    current = None
    for e in results["strata"]:
        if e["group"] != current:
            current = e["group"]
            lines.append(f"| **{current}** | | | | | | | | | | |")
        s, d = e["systems"], e["diffs"]
        small = " (n<30)" if e["n_instruments"] < 30 else ""
        lines.append(f"| {e['stratum']}{small} | {e['n_instruments']} | {e['n_frames']} | {cell(s['tapis'], 'acc')} | {cell(s['ours_single'], 'acc')} | {cell(s['ours_final'], 'acc')} | "
                     f"{dcell(d['ours_final-tapis'], 'acc')} | {dcell(d['ours_final-ours_single'], 'acc')} | {cell(s['tapis'], 'f1')} | {cell(s['ours_final'], 'f1')} | {dcell(d['ours_final-tapis'], 'f1')} |")
    (args.out_dir / "strata_table.md").write_text("\n".join(lines) + "\n")

    with open(args.out_dir / "per_instrument.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["instrument", "file", "case", "gt", "box_x", "box_y", "box_w", "box_h", "tapis", "single_s42", "single_s43", "single_s44", "final_s42", "final_s43", "final_s44"] + list(flags))
        for i in range(n):
            wr.writerow([i, te["file"][i], te["case"][i], y[i], int(te["box_x"][i]), int(te["box_y"][i]), int(te["box_w"][i]), int(te["box_h"][i]), tapis[i]]
                        + [ours["single"][s][i] for s in SEEDS] + [ours["final"][s][i] for s in SEEDS] + [int(flags[k][i]) for k in flags])
    print((args.out_dir / "strata_table.md").read_text())


if __name__ == "__main__":
    main()
