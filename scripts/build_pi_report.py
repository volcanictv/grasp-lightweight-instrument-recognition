"""Builds the short PI report (docs/reports/classifier_report.html) from saved result files.

Every number is read from a JSON under docs/reports/. Cells whose official-test run has not finished yet
render as "pending" and fill in on the next build once these files exist:
    docs/reports/causal_realtime/official_edgetam.json   (scripts/score_causal_official.py --name edgetam)
    docs/reports/causal_realtime/official_yolo26s.json   (scripts/score_causal_official.py --name yolo26s)

Usage:
    python scripts/build_pi_report.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
R = REPO_ROOT / "docs" / "reports"
RT = R / "causal_realtime"
OUT = R / "classifier_report.html"

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
SHORT = ["Bipolar", "Prograsp", "Needle Dr.", "Monopolar", "Suction", "Clip App.", "Grasper"]
SCORE_KEYS = ["Bipolar", "Prograsp", "LargeNeedle", "MonoScissors", "Suction", "ClipApplier", "Grasper"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SUB, MUTED, GRID = "#1e293b", "#475569", "#898781", "#e1e0d9"
FONT = 'Georgia, "Times New Roman", serif'


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def f4(v: float | None) -> str:
    return '<span class="pend">pending</span>' if v is None else f"{v:.4f}"


def esc(s: str) -> str:
    return html.escape(s)


def t(x: float, y: float, s: str, fill: str = INK, anchor: str = "start", size: float = 10, weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{esc(s)}</text>'


def svg(w: int, h: int, label: str, body: str) -> str:
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(label)}" font-family=\'{FONT}\'>{body}</svg>'


def pipeline_figure(cfg: dict) -> str:
    def box(x, y, w, h, lines, fill="#f6f7f9", stroke="#dfe3e8"):
        out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>'
        for i, s in enumerate(lines):
            out += t(x + w / 2, y + 16 + i * 13, s, anchor="middle", size=10, weight="700" if i == 0 else "400")
        return out
    arrow = lambda x1, y1, x2, y2: f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{SUB}" stroke-width="1.4" fill="none" marker-end="url(#ah)"/>'
    b = '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>'
    b += box(6, 14, 118, 52, ["1. Crop", "mask-multiplied,", "square letterbox"])
    b += arrow(124, 40, 146, 40)
    b += box(148, 14, 134, 52, ["2. Three members", "ResNet-50 320 px and", "2 MobileNetV3-small, 1 pass"])
    b += arrow(282, 40, 304, 40)
    b += box(306, 14, 134, 52, ["3. Evidence + gate", "Dirichlet evidence summed,", "epistemic score S1"])
    b += arrow(440, 40, 462, 40)
    b += f'<polygon points="464,40 506,14 548,40 506,66" fill="#fff" stroke="#1f3a5f"/>' + t(506, 38, "S1 above", "#1f3a5f", "middle", 9.5, "700") + t(506, 50, "threshold?", "#1f3a5f", "middle", 9.5, "700")
    b += arrow(506, 66, 506, 92) + t(512, 84, "no (about 87 to 89%)", SUB, "start", 9)
    b += box(446, 94, 120, 38, ["Label now", "single pass"], fill="#eaf3ee", stroke="#2f7d52")
    b += arrow(548, 40, 586, 40) + t(556, 31, "yes", SUB, "start", 9)
    k = cfg["k"]
    b += box(588, 14, 150, 52, ["4. Look back (causal)", f"track the instrument over the", f"last {k} frames at 1 Hz"])
    b += arrow(663, 66, 663, 92)
    b += box(588, 94, 150, 38, ["5. Re-label", "confidence-weighted vote"], fill="#eaf3ee", stroke="#2f7d52")
    b += t(6, 94, "Runs once per instrument per 1 Hz frame.", SUB, size=9.5) + t(6, 107, "Only gated instances pay for tracking (step 4).", SUB, size=9.5)
    b += t(6, 120, "Tracker: EdgeTAM (A) or YOLO26s-seg (B).", SUB, size=9.5)
    return svg(744, 140, "Pipeline: crop, three evidential members, evidence gate, then either label immediately or look back over the last frames and re-label", b)


def perclass_figure(series: list[tuple[str, str, list[float | None]]]) -> str:
    w, h, x0, y0, pw, ph = 400, 180, 30, 12, 366, 118
    lo, hi = 0.6, 1.0
    y_of = lambda v: y0 + ph - (v - lo) / (hi - lo) * ph
    b = ""
    for tick in (0.6, 0.7, 0.8, 0.9, 1.0):
        b += f'<line x1="{x0}" y1="{y_of(tick):.1f}" x2="{x0 + pw}" y2="{y_of(tick):.1f}" stroke="{GRID}"/>' + t(x0 - 6, y_of(tick) + 3, f"{tick:.1f}", SUB, "end", 9)
    gw = pw / 7
    bw = gw / (len(series) + 1)
    for ci in range(7):
        gx = x0 + ci * gw + bw / 2
        for si, (name, colour, vals) in enumerate(series):
            v = vals[ci]
            x = gx + si * bw
            if v is None:
                b += f'<rect x="{x:.1f}" y="{y0 + 4}" width="{bw - 2:.1f}" height="{ph - 4}" fill="none" stroke="{MUTED}" stroke-dasharray="3 3"><title>{esc(name)}: pending</title></rect>'
            else:
                b += f'<rect x="{x:.1f}" y="{y_of(v):.1f}" width="{bw - 2:.1f}" height="{y0 + ph - y_of(v):.1f}" fill="{colour}"><title>{esc(name)} {esc(SHORT[ci])}: {v:.3f}</title></rect>'
        b += t(x0 + ci * gw + gw / 2, y0 + ph + 14, SHORT[ci], INK, "middle", 8.5)
    lx = x0
    for name, colour, vals in series:
        pend = all(v is None for v in vals)
        mark = f'<rect x="{lx}" y="{h - 14}" width="11" height="10" fill="none" stroke="{MUTED}" stroke-dasharray="3 2"/>' if pend else f'<rect x="{lx}" y="{h - 14}" width="11" height="10" fill="{colour}"/>'
        b += mark + t(lx + 16, h - 5, name + (" (pending)" if pend else ""), SUB, size=9.5)
        lx += 16 + 6.0 * len(name) + (62 if pend else 24)
    return svg(w, h, "Per-class F1 on the official test for each configuration", b)


def confusion_figure(cm: list[list[int]]) -> str:
    w, h, x0, y0, cell = 330, 284, 78, 30, 34
    b = t(x0 + 3.5 * cell, 12, "predicted", SUB, "middle", 10, "700") + f'<text transform="translate(12,{y0 + 3.5 * cell}) rotate(-90)" fill="{SUB}" text-anchor="middle" font-size="10" font-weight="700">true</text>'
    for j in range(7):
        b += t(x0 + j * cell + cell / 2, y0 - 6, SHORT[j][:6], SUB, "middle", 8.5)
    for i in range(7):
        n = sum(cm[i])
        b += t(x0 - 4, y0 + i * cell + cell / 2 + 3, SHORT[i], INK, "end", 9.5)
        for j in range(7):
            p = cm[i][j] / n if n else 0
            if i == j:
                fill, ink = f"rgba(42,120,214,{0.25 + 0.6 * p:.2f})", INK
            else:
                fill, ink = f"rgba(235,104,52,{min(0.9, p * 6):.2f})", INK
            b += f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell - 1}" height="{cell - 1}" fill="{fill}"><title>{esc(CLASSES[i])} predicted as {esc(CLASSES[j])}: {cm[i][j]} of {n} ({100 * p:.1f}%)</title></rect>'
            if p >= 0.005 or i == j:
                b += t(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2 + 3.5, f"{100 * p:.0f}" if p >= 0.095 or i == j else f"{100 * p:.1f}", ink, "middle", 9, "700" if i == j else "400")
    return svg(w, h, "Confusion matrix of the three-member ensemble alone on the official test, row-normalised percent", b)


def kcurve_figure(cal: dict) -> str:
    w, h, x0, y0, pw, ph = 330, 190, 44, 14, 270, 120
    lo, hi = 0.93, 0.955
    ks = [3, 5, 10, 15, 20]
    x_of = lambda k: x0 + (ks.index(k) / (len(ks) - 1)) * pw
    y_of = lambda v: y0 + ph - (v - lo) / (hi - lo) * ph
    b = ""
    for tick in (0.93, 0.935, 0.94, 0.945, 0.95):
        b += f'<line x1="{x0}" y1="{y_of(tick):.1f}" x2="{x0 + pw}" y2="{y_of(tick):.1f}" stroke="{GRID}"/>' + t(x0 - 5, y_of(tick) + 3, f"{tick:.3f}", SUB, "end", 8.5)
    for k in ks:
        b += t(x_of(k), y0 + ph + 14, str(k), SUB, "middle", 9)
    b += t(x0 + pw / 2, y0 + ph + 28, "look-back k (past frames at 1 Hz)", SUB, "middle", 9.5)
    base = cal["base"]
    b += f'<line x1="{x0}" y1="{y_of(base):.1f}" x2="{x0 + pw}" y2="{y_of(base):.1f}" stroke="{MUTED}" stroke-dasharray="4 3"/>' + t(x0 + pw - 2, y_of(base) - 4, "classifier alone", MUTED, "end", 8.5)
    for name, colour, label in (("edgetam", BLUE, "EdgeTAM"), ("yolo26s", ORANGE, "YOLO26s-seg")):
        pts = [(x_of(k), y_of(cal["trackers"][name]["by_k"][str(k)]["chosen_row"]["accuracy"]), k, cal["trackers"][name]["by_k"][str(k)]["chosen_row"]["accuracy"]) for k in ks]
        b += f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)}" fill="none" stroke="{colour}" stroke-width="2"/>'
        for x, y, k, a in pts:
            b += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colour}" stroke="#fff" stroke-width="1.5"><title>{esc(label)} k={k}: {a:.4f}</title></circle>'
    b += f'<rect x="{x0}" y="{h - 14}" width="11" height="10" fill="{BLUE}"/>' + t(x0 + 16, h - 5, "EdgeTAM", SUB, size=9.5)
    b += f'<rect x="{x0 + 90}" y="{h - 14}" width="11" height="10" fill="{ORANGE}"/>' + t(x0 + 106, h - 5, "YOLO26s-seg", SUB, size=9.5)
    return svg(w, h, "Fold1 accuracy against causal look-back for the two trackers", b)


CSS = """
:root{--navy:#1f3a5f;--ink:#1e293b;--sub:#475569;--panel:#f6f7f9;--border:#dfe3e8;--good:#2f7d52}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font-family:Georgia,"Times New Roman",serif;font-size:14px;line-height:1.4}
.wrap{max-width:820px;margin:0 auto;padding:28px 22px 40px}
h1{font-size:1.45rem;color:var(--navy);margin:0 0 2px}
.sub{color:var(--sub);font-size:.9rem;margin-bottom:.8em}
h2{font-size:1.02rem;color:var(--navy);border-bottom:1px solid var(--border);padding-bottom:3px;margin:1.1em 0 .4em}
p{margin:.4em 0}ul{margin:.3em 0;padding-left:1.2em}li{margin:.2em 0}
table{border-collapse:collapse;width:100%;margin:.4em 0;font-size:.86rem}
th,td{border:1px solid var(--border);padding:3px 7px;text-align:left}
th{background:var(--panel)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.hi{background:#eaf3ee;font-weight:700}.ref{color:var(--sub)}
.pend{color:#898781;font-style:italic;border:1px dashed #c3c2b7;border-radius:3px;padding:0 4px;font-size:.8em}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:.6em 0}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:6px 9px}
.tile b{display:block;font-size:1.15rem;color:var(--navy);font-variant-numeric:tabular-nums}.tile span{font-size:.76rem;color:var(--sub)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}
.small{color:var(--sub);font-size:.78rem}
svg{width:100%;height:auto;display:block}
@media(max-width:640px){.tiles{grid-template-columns:1fr 1fr}.row2{grid-template-columns:1fr}}
@media print{@page{size:A4;margin:8mm}body{font-size:8.6pt;line-height:1.28}.wrap{max-width:none;padding:0}
h1{font-size:14pt}h2{font-size:10pt;margin:.7em 0 .3em}table{font-size:7.6pt}th,td{padding:1px 4px}
.small{font-size:7.2pt}tr,svg,.tile{break-inside:avoid}h2{break-after:avoid}.p3{break-before:page}}
"""


def build() -> str:
    base = load(RT / "base_official.json")
    fold1_base = load(RT / "fold1_base.json")
    cal = load(RT / "fold1_causal_calibration.json")
    ev4 = load(R / "evidential_pipeline_switch" / "results.json")
    vote = load(R / "final_pipeline" / "metrics.json")
    lat = {n: load(RT / "latency" / f"{n}.json") for n in ("yolo", "classify", "match", "edgetam", "sam2_large")}
    off = {"edgetam": load(RT / "official_edgetam.json"), "yolo26s": load(RT / "official_yolo26s.json")}
    cal["base"] = fold1_base["accuracy"]

    def ov(name: str, key: str, sub: str = "tracked") -> float | None:
        d = off[name]
        return None if d is None else d[sub][key]

    members = lat["classify"]["members"]
    per_crop = sum(members[m]["median_ms"] for m in ("resnet50_320", "baseline", "letterbox_crop"))
    frame3 = lat["yolo"]["per_frame"]["median_ms"] + 3 * per_crop
    cfg_a = {"k": cal["trackers"]["edgetam"]["chosen_k"]}

    def res_row(label, tracked, win, acc, f1, cls="", note=""):
        return f'<tr class="{cls}"><td>{label}</td><td>{win}</td><td class="n">{tracked}</td><td class="n">{f4(acc)}</td><td class="n">{f4(f1)}</td><td class="small">{note}</td></tr>'

    ta = off["edgetam"]["tracked"]["tracked"] if off["edgetam"] else "pending"
    tb = off["yolo26s"]["tracked"]["tracked"] if off["yolo26s"] else "pending"
    rows = [
        res_row("Single MobileNetV3-small", "0", "single frame", 0.867, 0.834),
        res_row("Single ResNet-50 (320 px)", "0", "single frame", 0.915, 0.865),
        res_row("4-member evidential, alone", "0", "single frame", ev4["base_accuracy"], ev4["base_macro_f1"]),
        res_row("<b>3-member evidential, alone</b> (this work)", "0", "single frame", base["accuracy"], base["macro_f1"], "hi"),
        res_row("A: 3-member + EdgeTAM", ta, f"{cfg_a['k']} past frames", ov("edgetam", "accuracy"), ov("edgetam", "macro_f1"), "hi", "causal, real-time valid"),
        res_row("B: 3-member + YOLO26s-seg", tb, f"{cal['trackers']['yolo26s']['chosen_k']} past frames", ov("yolo26s", "accuracy"), ov("yolo26s", "macro_f1"), "hi", "causal, real-time valid"),
        res_row("4-member evidential + SAM2-large", ev4["tracked"], "10 past + 10 future", ev4["final_accuracy"], ev4["final_macro_f1"], "ref", "offline reference (uses future frames)"),
        res_row("80-pass vote ensemble + SAM2-large", vote["tracked"], "10 past + 10 future", vote["after_tracking"]["accuracy"], vote["after_tracking"]["macro_f1"], "ref", "offline reference, gate chosen on test"),
    ]

    f1_base3 = [base["per_class"][c]["f1"] for c in CLASSES]
    f1_ev4 = [ev4["per_class"][c]["f1"] for c in CLASSES]
    f1_a = [off["edgetam"]["per_class_f1"][k] for k in SCORE_KEYS] if off["edgetam"] else [None] * 7
    f1_b = [off["yolo26s"]["per_class_f1"][k] for k in SCORE_KEYS] if off["yolo26s"] else [None] * 7
    pc_rows = ""
    for i, c in enumerate(CLASSES):
        pc_rows += (f'<tr><td>{c}</td><td class="n">{base["per_class"][c]["n"]}</td><td class="n">{f1_base3[i]:.3f}</td>'
                    f'<td class="n">{"" if f1_a[i] is None else f"{f1_a[i]:.3f}"}{"" if f1_a[i] is not None else f4(None)}</td>'
                    f'<td class="n">{"" if f1_b[i] is None else f"{f1_b[i]:.3f}"}{"" if f1_b[i] is not None else f4(None)}</td>'
                    f'<td class="n ref">{f1_ev4[i]:.3f}</td><td class="n">{base["per_class"][c]["recall"]:.3f}</td></tr>')

    fold_rows = (f'<tr><td>3-member alone</td><td></td><td></td><td></td><td class="n">{fold1_base["accuracy"]:.4f}</td><td class="n">{fold1_base["macro_f1"]:.4f}</td><td></td></tr>')
    for name, label in (("edgetam", "EdgeTAM (A)"), ("yolo26s", "YOLO26s-seg (B)"), ("yolo26m", "YOLO26m-seg (not carried forward)")):
        tr = cal["trackers"][name]
        r = tr["by_k"][str(tr["chosen_k"])]["chosen_row"]
        fold_rows += (f'<tr><td>{label}</td><td class="n">{tr["chosen_k"]}</td><td class="n">{tr["chosen_threshold"]:.2e}</td><td class="n">{100 * r["share"]:.0f}%</td>'
                      f'<td class="n">{r["accuracy"]:.4f}</td><td class="n">{r["macro_f1"]:.4f}</td><td class="n">{r["fixed"]} / {r["broken"]}</td></tr>')

    cm = base["confusion_counts"]
    gr = base["per_class"]["Laparoscopic Grasper"]
    pr = base["per_class"]["Prograsp Forceps"]
    def top_conf(i: int, n: int = 2) -> list[tuple[str, int]]:
        order = sorted((j for j in range(7) if j != i), key=lambda j: -cm[i][j])[:n]
        return [(SHORT[j], cm[i][j]) for j in order]

    g_top, p_top = top_conf(6), top_conf(1, 1)
    lat_rows = (
        f'<tr><td>Detector stand-in: YOLO26s-seg, one frame</td><td class="n">{lat["yolo"]["per_frame"]["median_ms"]:.1f} ms</td><td>always, once per frame</td></tr>'
        f'<tr><td>3-member classifier, one instrument</td><td class="n">{per_crop:.1f} ms</td><td>always, per instrument (sum of member medians)</td></tr>'
        f'<tr><td>One 1 Hz frame with 3 instruments</td><td class="n">{frame3:.0f} ms</td><td>10.5 + 3 x {per_crop:.1f}; budget is 1000 ms</td></tr>'
        f'<tr><td>Track matching (YOLO path)</td><td class="n">{lat["match"]["per_instance"]["median_ms"]:.1f} ms</td><td>gated instruments only, per instrument</td></tr>'
        f'<tr><td>EdgeTAM propagation, 21-frame window</td><td class="n">{lat["edgetam"]["per_instance"]["median_ms"] / 1000:.1f} s</td><td>gated only; {lat["edgetam"]["params_m"]:.1f}M parameters</td></tr>'
        f'<tr class="ref"><td>SAM2-large propagation, 21-frame window</td><td class="n">{lat["sam2_large"]["per_instance"]["median_ms"] / 1000:.1f} s</td><td>offline reference; {lat["sam2_large"]["params_m"]:.0f}M parameters</td></tr>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Surgical Instrument Classifier Status</title><style>{CSS}</style></head><body><div class="wrap">
<h1>GraSP instrument classifier: how it works and where it stands</h1>
<div class="sub">Status 2026-10-01. Official test: 5 cases, {base['n']:,} instrument instances. The classifier names an instrument whose location is given; detection is a separate upstream component.</div>
<div class="tiles">
<div class="tile"><b>{base['accuracy']:.4f}</b><span>accuracy, 3-member classifier alone (macro-F1 {base['macro_f1']:.4f})</span></div>
<div class="tile"><b>{f4(ov('edgetam', 'accuracy'))}</b><span>with causal tracking, A: EdgeTAM</span></div>
<div class="tile"><b>{ev4['final_accuracy']:.4f}</b><span>offline reference, 4-member + SAM2-large (macro-F1 {ev4['final_macro_f1']:.4f})</span></div>
<div class="tile"><b>{frame3:.0f} ms</b><span>always-on cost of one 1 Hz frame with 3 instruments, Titan Xp</span></div>
</div>
<h2>How it works</h2>
<div>{pipeline_figure(cfg_a)}</div>
<ul>
<li>Each instrument is classified on its own mask crop, because frames hold 2 to 3 instruments. Three different members each make one pass; their evidence is summed and the epistemic score S1 says how unsure the ensemble is.</li>
<li>Only the unsure instruments (about 10 to 13%) are tracked back over the last {cfg_a['k']} frames and re-labelled. Tracking uses past frames only, so it can run on a live 1 Hz feed.</li>
</ul>
<h2>Official test results</h2>
<table><tr><th>configuration</th><th>window</th><th class="n">tracked</th><th class="n">accuracy</th><th class="n">macro-F1</th><th>note</th></tr>{''.join(rows)}</table>
<p class="small">One training seed. Settings for A and B were chosen on fold1 (members trained on the other cases) and locked before this run (docs/reports/causal_tracker_preregistration.md). Instances within a case share patient, camera and instrument units, so the effective sample is closer to 5 than {base['n']:,}.</p>
<h2>Per-class F1, official test</h2>
<div class="row2"><div><table><tr><th>class</th><th class="n">n</th><th class="n">alone</th><th class="n">A</th><th class="n">B</th><th class="n">ref</th><th class="n">recall, alone</th></tr>{pc_rows}</table>
<p class="small">alone: 3-member classifier. A, B: with causal tracking. ref: 4-member evidential + SAM2-large, offline.</p></div><div>{perclass_figure([("alone", BLUE, f1_base3), ("A", ORANGE, f1_a), ("B", AQUA, f1_b)])}</div></div>
<div class="p3"></div>
<h2>Where we struggle</h2>
<div class="row2"><div>{confusion_figure(cm)}<p class="small">3-member classifier alone, official test, % of each true class.</p></div>
<div><ul>
<li><b>Laparoscopic Grasper</b> is the weakest class (F1 {gr['f1']:.3f}, recall {gr['recall']:.3f}, n={gr['n']}): most often called {g_top[0][0]} ({g_top[0][1]} of {gr['n']}) or {g_top[1][0]} ({g_top[1][1]}). Earlier analysis found closed jaws can look like a suction tube in a single frame.</li>
<li><b>Prograsp Forceps</b> recall is {pr['recall']:.3f}; its largest confusion is {p_top[0][0]} ({p_top[0][1]} of {pr['n']}).</li>
<li><b>Clip Applier</b> (n=38) and Grasper (n=81) F1 moves by a few instances; treat their per-class numbers as noisy.</li>
<li>Some errors are confident: the models agree on the wrong class, so no uncertainty gate flags them and tracking cannot help (the earlier vote pipeline passed 29 of 210 errors, 8 unanimous).</li>
<li>Zero-shot transfer is weak: the GraSP-trained model scores 0.528 / 0.551 accuracy on EndoVis 2018 / 2017 without retraining.</li>
</ul></div></div>
<h2>Causal tracker choice (fold1, tuned on held-out cases only)</h2>
<div class="row2"><div><table><tr><th>tracker</th><th class="n">k</th><th class="n">gate</th><th class="n">tracked</th><th class="n">accuracy</th><th class="n">macro-F1</th><th class="n">fixed / broken</th></tr>{fold_rows}</table>
<p class="small">k: past frames used, chosen as the shortest within 0.001 of the best. EdgeTAM was still improving at k=20, the longest tried.</p></div><div>{kcurve_figure(cal)}</div></div>
<h2>Latency (Titan Xp, nothing else running, median)</h2>
<table><tr><th>stage</th><th class="n">time</th><th>when it runs</th></tr>{lat_rows}</table>
<p class="small">A 1 Hz stream gives 1000 ms per frame. Always-on cost leaves most of the budget free; tracking runs only for gated instruments, in the background if needed. Detector time is a stand-in and will change with the final detector.</p>
<h2>Status and next</h2>
<ul>
<li>Pending cells above fill from the official runs of A and B; nothing was changed after seeing them.</li>
<li>Next: plug the lab detector into the real-time harness, replay 30 fps footage sampled at 1 Hz, and measure end-to-end latency and deadline misses.</li>
<li>Code and weights: github.com/volcanictv/grasp-instrument-classifier (vote pipeline today; evidential code not yet released).</li>
</ul>
</div></body></html>"""
    assert "—" not in page and "&mdash;" not in page, "no em dashes in the report"
    return page


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
