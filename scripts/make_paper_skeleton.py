"""Pre-populated fields of the paper skeleton, generated from the result files (nothing typed by hand): numbers.tex (one \\newcommand per locked number), tables/*.tex and figures/*.pdf.

Inputs (all 3-seed means on the five official test cases unless stated): docs/reports/gtbox_sam/{bootstrap_heldout_row,tta_vs_tracking,gate_baselines,baselines_on_test,final_3seed,final_static_noncausal_3seed}.json,
docs/reports/paper_gflops.json, docs/reports/refine_all_analysis.json, reviewer_check_window.json (window sweep).

    python scripts/make_paper_skeleton.py --evidence C:/Users/aryan/paper_sandbox/evidence --out C:/Users/aryan/paper_sandbox/paper_v5_skeleton
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42,
         "text.color": "#1c2430", "axes.edgecolor": "#8a94a3", "axes.labelcolor": "#1c2430", "xtick.color": "#5d6877", "ytick.color": "#1c2430"}
plt.rcParams.update(STYLE)
INK, MUTED, ACCENT, GREY, AMBER, RED, TEAL = "#1c2430", "#5d6877", "#0b6e8a", "#8a94a3", "#a8651a", "#b3382f", "#2a9d8f"
CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def f2(x: float) -> str:
    return f"{x:.2f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    E, O = args.evidence, args.out
    (O / "tables").mkdir(parents=True, exist_ok=True)
    (O / "figures").mkdir(parents=True, exist_ok=True)
    J = lambda rel: json.loads((E / rel).read_text(encoding="utf-8"))
    BHR, TTA, GB, BT = J("gtbox_sam/bootstrap_heldout_row.json"), J("gtbox_sam/tta_vs_tracking.json"), J("gtbox_sam/gate_baselines.json"), J("gtbox_sam/baselines_on_test.json")
    F3S, STAT, GF, RA = J("gtbox_sam/final_3seed.json"), J("gtbox_sam/final_static_noncausal_3seed.json"), J("paper_gflops.json"), J("refine_all_analysis.json")
    WIN = J("titanxp_extra/reviewer_check_window.json")
    seeds = sorted(GB["seeds"])
    est = BHR["estimates"]
    CFG = {"single": "single pass", "tau": "gated (tau 1.7e-05)", "all": "gated (top 2861)"}
    M = ("mIoU", "IoU", "mcIoU")
    num: dict[str, str] = {}

    def put(name: str, value: str, comment: str = "") -> None:
        assert not any(ch.isdigit() for ch in name), f"TeX macro names cannot contain digits: {name}"
        num[name] = f"\\newcommand{{\\{name}}}{{{value}}}" + (f" % {comment}" if comment else "")

    SHARE_WORD = {10: "Ten", 20: "Twenty", 30: "Thirty", 45: "FortyFive", 60: "Sixty", 100: "Hundred"}

    # ---- Table 1 numbers ----
    acc = {"single": F3S["configs"]["single pass"]["instance_accuracy"]["mean"], "tau": F3S["configs"]["gated (tau 1.7e-05)"]["instance_accuracy"]["mean"],
           "all": float(np.mean([RA["seeds"][s]["accuracy_all_refined"] for s in RA["seeds"]]))}
    rows1 = []
    for key, label in (("single", "Single pass"), ("tau", r"Gated temporal refinement, $\tau=1.7\times10^{-5}$"), ("all", "Temporal refinement of every instrument")):
        e = est[CFG[key]]["three_seed_mean"]
        cells = []
        for m in M:
            p, (lo, hi) = e[m]["point"], e[m]["ci95_case"]
            put(f"num{key.capitalize()}{m}", f2(p), f"3-seed mean, bootstrap_heldout_row.json")
            put(f"num{key.capitalize()}{m}Lo", f2(lo))
            put(f"num{key.capitalize()}{m}Hi", f2(hi))
            cells.append(f"{f2(p)} [{f2(lo)}, {f2(hi)}]")
        put(f"num{key.capitalize()}Acc", f"{100 * acc[key]:.2f}", "instance accuracy, percent")
        rows1.append(f"{label} & " + " & ".join(cells) + f" & {100 * acc[key]:.2f}\\% \\\\")
    tab1 = ("\\begin{table}[tbp]\n\\floatconts\n  {tab:main}\n  {\\caption{Main results on the GraSP test cases (5 cases, 1,125 frames, 2,861 instruments), mean over three classifier seeds; 95\\% case-bootstrap intervals in brackets. "
            "All results use oracle ground-truth boxes. The gated threshold was selected during development on fold~1 with an earlier pipeline whose classifiers were trained on fold~2, and then applied unchanged to the final pipeline.}}\n"
            "  {\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lrrrr}\n  \\toprule\n  \\bfseries Method & \\bfseries mIoU & \\bfseries IoU & \\bfseries mcIoU & \\bfseries Inst.\\ acc.\\\\\n  \\midrule\n  "
            + "\n  ".join(rows1) + "\n  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n")
    (O / "tables" / "tab_main.tex").write_text(tab1, encoding="utf-8")
    # paired differences
    pd = BHR["paired_differences"]
    k_sg, k_ga = "gated (tau 1.7e-05) minus single pass", "gated (tau 1.7e-05) minus gated (top 2861)"
    k_as = "gated (top 2861) minus single pass"
    for tag, key in (("GatedMinusSingle", k_sg), ("GatedMinusAll", k_ga), ("AllMinusSingle", k_as)):
        for m in M:
            d = pd[key][m]
            put(f"num{tag}{m}", f"{d['point']:+.2f}".replace("+", "+"), "paired difference, case bootstrap")
            put(f"num{tag}{m}Lo", f"{d['ci95_case'][0]:+.2f}")
            put(f"num{tag}{m}Hi", f"{d['ci95_case'][1]:+.2f}")

    # ---- Table 2: compute ----
    av = GF["average"]
    comp = [("Single pass", "0\\%", GF["single_pass_per_instrument"], None),
            ("Selective refinement", "29\\%", av["29% refined"]["gflops"], None), ("Gated operating point", "45\\%", av["held-out threshold, 45% refined"]["gflops"], None),
            ("Temporal refinement of every instrument", "100\\%", av["every instrument refined"]["gflops"], None)]
    full = comp[-1][2]
    rows2 = []
    for i, (name, share, g, _x) in enumerate(comp):
        sav = "--" if i == 0 else f"{100 * (1 - g / full):.1f}\\%"
        rows2.append(f"{name} & {share} & {g:,.1f} & {sav} \\\\".replace(",", "{,}"))
        put(f"numGflops{['Single', 'ShareGate', 'Gated', 'All'][i]}", f"{g:,.1f}".replace(",", "{,}"), "GFLOPs per instrument, multiply-add convention")
        put(f"numSaving{['Single', 'ShareGate', 'Gated', 'All'][i]}", "0" if i == 0 else f"{100 * (1 - g / full):.1f}")
    put("numGpuFrameStage", f3(GF["a100_seconds"]["single_pass_per_frame"]), "seconds per frame on an A100 (job 21823404)")
    put("numGpuRefinement", f3(GF["a100_seconds"]["refinement_per_instrument"]), "seconds per refined instrument on an A100")
    tab2 = ("\\begin{table}[tbp]\n\\floatconts\n  {tab:compute}\n  {\\caption{Compute per instrument, averaged over all test instruments (GFLOPs, one multiply-add counted as one FLOP). On an A100 the single-pass frame stage took about "
            f"{GF['a100_seconds']['single_pass_per_frame']:.3f}\\,s and refinement about 6.087\\,s per refined instrument.}}}}\n"
            "  {\\footnotesize\\begin{tabular}{lrrr}\n  \\toprule\n  \\bfseries Policy & \\bfseries Refined share & \\bfseries GFLOPs/instrument & \\bfseries Saving vs.\\ refine-all\\\\\n  \\midrule\n  " + "\n  ".join(rows2)
            + "\n  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n").replace("}}}}", "}}")
    (O / "tables" / "tab_compute.tex").write_text(tab2, encoding="utf-8")

    # ---- TTA, static control ----
    P = TTA["point"]
    for k, nm in (("single", "Single"), ("TTA all", "TtaAll"), ("TTA at tau", "TtaTau"), ("tracking all", "TrackAll"), ("tracking at tau", "TrackTau")):
        put(f"numFigTwo{nm}", f"{P[k]['mIoU']:.3f}", "mIoU, tta_vs_tracking.json")
    d13 = TTA["paired"]["tracking at tau minus TTA at tau"]["mIoU"]
    put("numTrackMinusTtaTau", f"{d13['point']:+.2f}")
    put("numTrackMinusTtaTauLo", f"{d13['ci95_case'][0]:+.2f}")
    put("numTrackMinusTtaTauHi", f"{d13['ci95_case'][1]:+.2f}")
    put("numTtaGainAll", f"{P['TTA all']['mIoU'] - P['single']['mIoU']:+.2f}", "TTA minus single pass, mIoU")
    put("numTrackGainAll", f"{P['tracking all']['mIoU'] - P['single']['mIoU']:+.2f}")
    put("numTtaFractionG", f"{TTA['g']['on all instruments']['mIoU']['point']:.3f}", "g = TTA gain / tracking gain, all instruments")
    sn = STAT["configs"]["gated (tau 2.8e-04)"]["mIoU"]["mean"]
    put("numStaticMiou", f"{100 * sn:.3f}", "static non-causal control at its stored threshold tau = 2.8e-4 (a matched static result at tau 1.7e-5 does not exist)")

    # ---- Fig 1 budget curve numbers ----
    shares = [f"{int(round(100 * s))}%" for s in GB["shares"]]
    cur = lambda name: [float(np.mean([GB["seeds"][s]["curves"][name][k]["mIoU"] for s in seeds])) for k in shares]
    single_m = float(np.mean([GB["seeds"][s]["single"]["mIoU"] for s in seeds]))
    full_m = float(np.mean([GB["seeds"][s]["refine_all"]["mIoU"] for s in seeds]))
    lines = [("S", "Evidential $S_1$", ACCENT, "-", "o"), ("largest belief", "Evidential max-belief", TEAL, "-", "s"), ("softmax max-prob", "CE max-softmax", AMBER, "-", "^"), ("random", "Random", "#b0b7c3", ":", None)]
    # supplement table: all orderings
    names_all = ["S", "largest belief", "belief entropy", "softmax max-prob", "softmax entropy", "MC vote disagreement", "MC mutual information", "class prior", "small box", "border box", "random", "best possible"]
    label_all = {"S": "Evidential $S_1$", "largest belief": "Evidential max-belief", "belief entropy": "Evidential belief entropy", "softmax max-prob": "CE max-softmax", "softmax entropy": "CE softmax entropy",
                 "MC vote disagreement": "MC-dropout vote", "MC mutual information": "MC-dropout mutual information", "class prior": "Class prior (weak classes first)", "small box": "Small box first",
                 "border box": "Border box first", "random": "Random", "best possible": "Best possible ordering"}
    rows3 = [f"Single pass (0\\%) & \\multicolumn{{{len(shares)}}}{{c}}{{{f2(single_m)}}} \\\\", "\\midrule"]
    for n in names_all:
        v = cur(n)
        rows3.append(f"{label_all[n]} & " + " & ".join(f2(x) for x in v) + " \\\\")
    for nm, key in (("S", "numBudgetS"), ("largest belief", "numBudgetBelief"), ("softmax max-prob", "numBudgetSoftmax"), ("random", "numBudgetRandom"), ("best possible", "numBudgetBest")):
        for sh, x in zip(shares, cur(nm)):
            put(f"{key}{SHARE_WORD[int(sh.rstrip('%'))]}", f"{x:.3f}", "mIoU at that share of instruments refined, 3-seed mean, gate_baselines.json")
    tab3 = ("\\begin{table}[tbp]\n\\floatconts\n  {tab:budget}\n  {\\caption{Test mIoU when the given share of instruments (columns) is refined in the order of each score; 3-seed mean. The best possible ordering is an oracle (instruments whose label tracking fixes first).}}\n"
            "  {\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{l" + "r" * len(shares) + "}\n  \\toprule\n  \\bfseries Ordering & " + " & ".join(f"\\bfseries {s.replace('%', chr(92) + '%')}" for s in shares)
            + "\\\\\n  \\midrule\n  " + "\n  ".join(rows3) + "\n  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n")
    (O / "tables" / "tab_budget_supp.tex").write_text(tab3, encoding="utf-8")

    # ---- Fig 1 ----
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = [0] + [int(s.rstrip("%")) for s in shares]
    for name, label, color, ls, marker in lines:
        ax.plot(xs, [single_m] + cur(name), ls=ls, color=color, marker=marker, ms=3.5, lw=1.5 if name != "random" else 1.0, label=label)
    best = cur("best possible")
    ax.plot(xs[1:], best, ls="--", color=INK, lw=0.9, label="Best possible")
    ax.axhline(full_m, color=MUTED, lw=0.6)
    ax.text(101, full_m, f"refine all {full_m:.1f}", fontsize=6.5, color=MUTED, va="center")
    ax.axhline(single_m, color=MUTED, lw=0.6)
    ax.text(101, single_m, f"single pass {single_m:.1f}", fontsize=6.5, color=MUTED, va="center")
    ax.set_xlim(0, 100)
    ax.set_xlabel("fraction of instrument instances refined (%)")
    ax.set_ylabel("test mIoU")
    ax.grid(color="#e1e5ea", lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=6.5, frameon=False, ncol=3, columnspacing=1.2)
    fig.subplots_adjust(left=0.1, right=0.80, top=0.97, bottom=0.27)
    fig.savefig(O / "figures" / "fig1_budget.pdf")
    plt.close(fig)

    # ---- Fig 2 ----
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    bars = [("Single pass", P["single"]["mIoU"], GREY), ("TTA, 21 views\nof the keyframe", P["TTA at tau"]["mIoU"], AMBER), ("Temporal tracking,\nall instruments", P["tracking all"]["mIoU"], ACCENT), (r"Temporal tracking,"+"\n"+r"gated $\tau$", P["tracking at tau"]["mIoU"], TEAL)]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, color=c, width=0.62)
        ax.text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.bar(4.3, 100 * sn, color="#d9dde3", width=0.42, hatch="///", edgecolor=GREY)
    ax.text(4.3, 100 * sn + 0.25, f"{100 * sn:.2f}$^\\dagger$", ha="center", fontsize=7, color=INK)
    ax.set_xticks(list(range(4)) + [4.3])
    ax.set_xticklabels([b[0] for b in bars] + ["Static non-causal\ncontrol"], fontsize=6.3)
    ax.set_ylim(70, 90)
    ax.set_ylabel("test mIoU")
    ax.grid(axis="y", color="#e1e5ea", lw=0.5)
    ax.text(-0.35, 89.0, f"gated tracking minus TTA: +{d13['point']:.2f} mIoU, 95% CI [{d13['ci95_case'][0]:+.2f}, {d13['ci95_case'][1]:+.2f}]", fontsize=6.5, color=INK, va="center")
    fig.text(0.01, 0.01, r"$^\dagger$ static control at its stored threshold $\tau=2.8\times10^{-4}$; a matched result at $\tau=1.7\times10^{-5}$ is not available", fontsize=5.8, color=MUTED)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.97, bottom=0.25)
    fig.savefig(O / "figures" / "fig2_temporal.pdf")
    plt.close(fig)

    # ---- per-class ----
    pc_s, pc_g = F3S["configs"]["single pass"]["per_class_iou"], F3S["configs"]["gated (tau 1.7e-05)"]["per_class_iou"]
    get = lambda d, i: float(d[str(i)]["mean"] if isinstance(d[str(i)], dict) else d[str(i)]) * 100
    rows4, gains = [], []
    for i, c in enumerate(CLASSES, start=1):
        s, g = get(pc_s, i), get(pc_g, i)
        gains.append(g - s)
        letter = "ABCDEFG"[i - 1]
        put(f"numClass{letter}Single", f2(s), c)
        put(f"numClass{letter}Gated", f2(g))
        put(f"numClass{letter}Gain", f"{g - s:+.2f}")
        rows4.append(f"{c} & {f2(s)} & {f2(g)} & {g - s:+.2f} \\\\")
    tab4 = ("\\begin{table}[tbp]\n\\floatconts\n  {tab:perclass}\n  {\\caption{Per-class IoU on the test cases, single pass and gated temporal refinement at $\\tau=1.7\\times10^{-5}$ (3-seed mean); gain in points.}}\n"
            "  {\\footnotesize\\begin{tabular}{lrrr}\n  \\toprule\n  \\bfseries Class & \\bfseries Single & \\bfseries Gated $\\tau$ & \\bfseries Gain\\\\\n  \\midrule\n  " + "\n  ".join(rows4) + "\n  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n")
    (O / "tables" / "tab_perclass_supp.tex").write_text(tab4, encoding="utf-8")
    order = np.argsort([get(pc_s, i) for i in range(1, 8)])
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for r, idx in enumerate(order):
        s, g = get(pc_s, idx + 1), get(pc_g, idx + 1)
        ax.barh(r + 0.18, s, height=0.34, color="#c9ced6", label="Single pass" if r == 0 else None)
        ax.barh(r - 0.18, g, height=0.34, color=ACCENT, label=r"Gated $\tau$" if r == 0 else None)
        ax.text(max(s, g) + 0.8, r, f"+{g - s:.2f}", va="center", fontsize=6.8, color=INK, fontweight="bold" if g - s > 10 else "normal")
    ax.set_yticks(range(7))
    ax.set_yticklabels([CLASSES[i] for i in order], fontsize=6.8)
    ax.set_xlim(40, 105)
    ax.set_xlabel("class IoU (%)")
    ax.legend(loc="lower right", fontsize=6.5, frameon=False)
    ax.grid(axis="x", color="#e1e5ea", lw=0.5)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.98, bottom=0.15)
    fig.savefig(O / "figures" / "fig3_perclass.pdf")
    plt.close(fig)

    # ---- Supplement: uncertainty baselines on the final test crops ----
    rows5 = []
    for name, e in BT["methods"].items():
        for sig, v in e["signals"].items():
            rows5.append(f"{name}: {sig} & {e['networks']} & {e['calibration']['accuracy']:.3f} & {e['calibration']['ece']:.3f} & {v['auroc']:.3f} & {100 * v['caught20']:.1f}\\% \\\\")
    S1 = BT["methods"]["Evidential (1 pass)"]["signals"]["epistemic S1"]
    mb = BT["methods"]["Evidential (1 pass)"]["signals"]["max-belief"]
    ms = BT["methods"]["Softmax ensemble"]["signals"]["max-softmax"]
    for nm, v in (("S", S1), ("Belief", mb), ("Softmax", ms)):
        put(f"numAuroc{nm}", f3(v["auroc"]), "baselines_on_test.json")
        put(f"numCaught{nm}", f"{100 * v['caught20']:.1f}")
    for lab, key in (("Belief", "Evidential (1 pass): max-belief"), ("Softmax", "Softmax ensemble: max-softmax")):
        c = BT["paired_vs_epistemic_S1"][key]["caught20"]
        put(f"numPaired{lab}Caught", f"{100 * c['point']:+.1f}", "S1 minus other, errors found at 20%")
        put(f"numPaired{lab}CaughtLo", f"{100 * c['ci95_cases'][0]:+.1f}")
        put(f"numPaired{lab}CaughtHi", f"{100 * c['ci95_cases'][1]:+.1f}")
        a = BT["paired_vs_epistemic_S1"][key]["auroc"]
        put(f"numPaired{lab}Auroc", f"{a['point']:+.3f}")
        put(f"numPaired{lab}AurocLo", f"{a['ci95_cases'][0]:+.3f}")
        put(f"numPaired{lab}AurocHi", f"{a['ci95_cases'][1]:+.3f}")
    tab5 = ("\\begin{table}[tbp]\n\\floatconts\n  {tab:uncertainty}\n  {\\caption{Error detection on the final test crops (2,861 instruments; 3 seeds, the 12-network rows pool the seeds). AUROC: separation of wrong from right single-pass labels. "
            "Errors found: share of all errors among the 20\\% of instruments with the highest score. CE: cross-entropy members trained with the recipe of the evidential finals.}}\n"
            "  {\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lrrrrr}\n  \\toprule\n  \\bfseries Method: score & \\bfseries Nets & \\bfseries Acc. & \\bfseries ECE & \\bfseries AUROC & \\bfseries Found@20\\\\\n  \\midrule\n  "
            + "\n  ".join(rows5) + "\n  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n")
    (O / "tables" / "tab_uncertainty_supp.tex").write_text(tab5, encoding="utf-8")

    # ---- Supplement: window sweep ----
    rows6 = []
    for k in sorted(WIN, key=lambda x: int(x)):
        w = WIN[k]
        rows6.append(f"$\\pm{k}$ & {100 * w['mIoU']:.2f} & {100 * w['IoU']:.2f} & {100 * w['mcIoU']:.2f} \\\\")
    tab6 = ("\\begin{table}[tbp]\n\\floatconts\n  {tab:window}\n  {\\caption{Tracking window: only the frames within $k$ of the keyframe are fused, for the 833 instruments with the highest $S_1$ (a fixed budget), mean of three seeds; $k=0$ is the single pass.}}\n"
            "  {\\footnotesize\\begin{tabular}{lrrr}\n  \\toprule\n  \\bfseries Window & \\bfseries mIoU & \\bfseries IoU & \\bfseries mcIoU\\\\\n  \\midrule\n  " + "\n  ".join(rows6) + "\n  \\bottomrule\n  \\end{tabular}}\n\\end{table}\n")
    (O / "tables" / "tab_window_supp.tex").write_text(tab6, encoding="utf-8")

    # ---- numbers.tex ----
    header = ("% Locked numbers, generated by scripts/make_paper_skeleton.py from the result files. Use these macros in the text so a number is never typed twice.\n"
              "% Sources: bootstrap_heldout_row.json, tta_vs_tracking.json, gate_baselines.json, baselines_on_test.json, final_3seed.json, final_static_noncausal_3seed.json, paper_gflops.json.\n"
              "% Segmenter selection record (development fold 1, mean IoU): SAM2+flip 0.908575, SAM3+flip 0.907581, ensemble 0.913068, pre-specified bar 0.9136, stored verdict 'keep sam2', final addendum: ensemble adopted.\n")
    seg = {"numSegSamTwo": "0.908575", "numSegSamThree": "0.907581", "numSegEnsemble": "0.913068", "numSegBar": "0.9136"}
    for k, v in seg.items():
        put(k, v, "segmenter selection record, sam3_fold1.json / gtbox_sam_protocol.md")
    (O / "numbers.tex").write_text(header + "\n".join(num.values()) + "\n", encoding="utf-8")
    print("wrote", len(num), "macros,", len(list((O / 'tables').glob('*.tex'))), "tables,", len(list((O / 'figures').glob('*.pdf'))), "figures to", O)


if __name__ == "__main__":
    main()
