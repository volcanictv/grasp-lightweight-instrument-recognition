"""Inline-SVG figures for docs/reports/grasp_report_2026-09-01.html, generated
from the saved result files so every plotted number is traceable:

  votes      how one instrument's 80 votes become a prediction, an
             uncertainty and a tracking decision (an illustration, marked as such)
  roc        ROC curves of the uncertainty signals   (final_pipeline/roc_curves.json)
  sweep      accuracy, macro-F1 and tracked share vs the gate
                                                     (tracking_gate_sweep/results.json)
  perclass   per-class F1 before and after tracking  (final_pipeline/metrics.json)

Colours are the dataviz skill's categorical slots 1-3 (blue, orange, aqua),
validated together on a white surface; text uses ink tokens, never series colours.
Writes one .svg per figure into docs/reports/figures/ (the report inlines them).

Usage:
    python scripts/build_report_figures.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "docs" / "reports"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, SUB, MUTED, GRID, BASE, SURFACE = "#1e293b", "#475569", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def svg_open(w: int, h: int, label: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{html.escape(label)}" '
            f'font-family=\'{FONT}\' font-size="11">')


def text(x: float, y: float, s: str, fill: str = INK, anchor: str = "start", size: int = 11, weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{html.escape(s)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = GRID, width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def dot(x: float, y: float, fill: str, tip: str, r: float = 5) -> str:
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{SURFACE}" stroke-width="2">'
            f'<title>{html.escape(tip)}</title></circle>')


def legend(items: list[tuple[str, str, str]], x: float, y: float, gap: float = 20) -> str:
    """items: (label, colour, key[, dash]) with key 'line', 'box' or 'dot'; an
    optional 4th element is a stroke-dasharray for 'line' entries (e.g. "6 4"),
    the texture channel used to tell a 4th line series from the first three
    hues without adding a fourth risky categorical colour."""
    out, yy = [], y
    for item in items:
        label, colour, key = item[0], item[1], item[2]
        dash = item[3] if len(item) > 3 else ""
        if key == "dot":
            out.append(f'<circle cx="{x + 7}" cy="{yy - 4}" r="5" fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>')
        elif key == "line":
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<line x1="{x}" y1="{yy - 4}" x2="{x + 18}" y2="{yy - 4}" stroke="{colour}" stroke-width="2" stroke-linecap="round"{dash_attr}/>')
        else:
            out.append(f'<rect x="{x}" y="{yy - 10}" width="14" height="12" rx="3" fill="{colour}"/>')
        out.append(text(x + 24, yy, label, SUB))
        yy += gap
    return "".join(out)


def figure_votes() -> str:
    w, h = 760, 200
    bx, bw, by, bh = 250, 480, 78, 24
    win, dis = 68, 12
    gate = 0.09
    x_split = bx + bw * win / 80
    parts = [svg_open(w, h, "Example: 80 votes, 68 for the winning class and 12 against, a 15 percent disagreement, above the 9 percent gate, so the instrument is sent to tracking")]
    parts.append(text(10, 22, "Example: how one instrument is decided", INK, size=12, weight="600"))
    parts.append(text(10, 60, "4 members × 20 passes", INK, weight="600"))
    parts.append(text(10, 76, "= 80 votes, each for the", SUB))
    parts.append(text(10, 91, "class of its largest output", SUB))
    parts.append(f'<path d="M188,88 H240" stroke="{MUTED}" stroke-width="1.5" fill="none"/><path d="M240,88 l-7,-4 v8 z" fill="{MUTED}"/>')
    parts.append(f'<path d="M{bx},{by} h{x_split - bx - 2:.1f} v{bh} h-{x_split - bx - 2:.1f} z" fill="{BLUE}"><title>68 of 80 votes for the winning class</title></path>')
    parts.append(f'<path d="M{x_split:.1f},{by} h{bx + bw - x_split - 4:.1f} a4,4 0 0 1 4,4 v{bh - 8} a4,4 0 0 1 -4,4 h-{bx + bw - x_split - 4:.1f} z" fill="{ORANGE}"><title>12 of 80 votes for other classes</title></path>')
    parts.append(text(bx + 10, by + 16, "68 votes: the winning class (85%)", SURFACE, size=11, weight="600"))
    parts.append(text(x_split + 8, by + 16, "12 (15%)", INK, size=11, weight="600"))
    parts.append(text(bx, by - 12, "Prediction = the class with the most votes", INK, weight="600"))
    parts.append(text(bx + bw, by - 12, "Uncertainty = share of votes against it", INK, anchor="end", weight="600"))
    gx = bx + bw - bw * gate
    parts.append(line(gx, by + bh + 2, gx, by + bh + 12, INK, 1.5))
    parts.append(f'<path d="M{gx:.1f},{by + bh + 12} H{bx + bw - 2:.1f}" stroke="{INK}" stroke-width="1.5" fill="none"/>')
    parts.append(line(bx + bw - 2, by + bh + 2, bx + bw - 2, by + bh + 12, INK, 1.5))
    parts.append(text(gx - 8, by + bh + 16, "gate: 9% disagreement", INK, anchor="end", weight="600"))
    parts.append(f'<rect x="{bx}" y="{by + bh + 44}" width="{bw}" height="34" rx="6" fill="#eaf3ee" stroke="#2f7d52" stroke-width="1"/>')
    parts.append(text(bx + bw / 2, by + bh + 65, "15% of votes disagree, which is 9% or more: send this instrument to tracking", INK, anchor="middle", size=12, weight="600"))
    parts.append(legend([("votes for the winning class", BLUE, "box"), ("votes for any other class", ORANGE, "box")], 10, 150, 20))
    parts.append("</svg>")
    return "".join(parts)


def figure_roc() -> str:
    data = json.loads((REPORTS / "final_pipeline" / "roc_curves.json").read_text())["curves"]
    order = [("MC Dropout vote disagreement (main method)", "MC Dropout vote disagreement", BLUE, ""),
             ("Between-member disagreement (dropout off)", "Between-member disagreement (dropout off)", ORANGE, ""),
             ("Softmax confidence (baseline)", "Softmax confidence (baseline)", AQUA, ""),
             ("Evidential epistemic uncertainty (own prediction)", "Evidential epistemic uncertainty", YELLOW, "6 4")]
    w, h = 560, 450
    px0, py0, pw, ph = 66, 22, 470, 380
    parts = [svg_open(w, h, "ROC curves: how well each uncertainty signal separates wrong from correct predictions")]
    for i in range(6):
        v = i / 5
        x, y = px0 + pw * v, py0 + ph * (1 - v)
        parts.append(line(px0, y, px0 + pw, y))
        parts.append(line(x, py0, x, py0 + ph))
        parts.append(text(px0 - 8, y + 4, f"{v:.1f}", MUTED, anchor="end"))
        parts.append(text(x, py0 + ph + 16, f"{v:.1f}", MUTED, anchor="middle"))
    parts.append(line(px0, py0 + ph, px0 + pw, py0 + ph, BASE))
    parts.append(line(px0, py0, px0, py0 + ph, BASE))
    parts.append(line(px0, py0 + ph, px0 + pw, py0, MUTED, 1))
    parts.append(text(px0 + pw * 0.62, py0 + ph * 0.66, "chance", MUTED, size=10))
    for key, label, colour, dash in order:
        c = data[key.replace(" (main method)", "")]
        pts = " ".join(f"{px0 + pw * f:.1f},{py0 + ph * (1 - t):.1f}" for f, t in zip(c["fpr"], c["tpr"]))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash_attr}><title>{html.escape(label)}: AUROC {c["auroc"]:.3f}</title></polyline>')
    parts.append(text(px0 + pw / 2, py0 + ph + 40, "Share of correct predictions flagged (false alarms)", SUB, anchor="middle"))
    parts.append(f'<text transform="translate(16,{py0 + ph / 2}) rotate(-90)" fill="{SUB}" text-anchor="middle" font-size="11">Share of errors caught</text>')
    lx, ly = px0 + pw - 268, py0 + ph - 96
    parts.append(f'<rect x="{lx - 10}" y="{ly - 18}" width="270" height="98" rx="6" fill="{SURFACE}" stroke="{GRID}"/>')
    entries = []
    for key, label, colour, dash in order:
        c = data[key.replace(" (main method)", "")]
        entries.append((f"{label.split(' (')[0]}  AUROC {c['auroc']:.3f}", colour, "line", dash))
    parts.append(legend(entries, lx, ly, 22))
    parts.append("</svg>")
    return "".join(parts)


def figure_sweep() -> str:
    rows = json.loads((REPORTS / "tracking_gate_sweep" / "results.json").read_text())["rows"]
    gates = sorted((float(k) for k in rows), reverse=True)
    acc = [rows[str(g)]["accuracy"] for g in gates]
    f1 = [rows[str(g)]["macro_f1"] for g in gates]
    trk = [100 * rows[str(g)]["tracked"] / 2861 for g in gates]
    w, h = 720, 470
    px0, pw = 70, 560
    a0, ah = 34, 210
    b0, bh = 300, 100
    lo, hi = 0.925, 0.970

    def X(g: float) -> float:
        return px0 + pw * (0.20 - g) / (0.20 - 0.03)

    def Ya(v: float) -> float:
        return a0 + ah * (hi - v) / (hi - lo)

    def Yb(v: float) -> float:
        return b0 + bh * (40 - v) / 40

    parts = [svg_open(w, h, "Accuracy and macro-F1, and the share of instances tracked, as the tracking gate is lowered from 20 percent to 3 percent")]
    parts.append(text(px0, 16, "Accuracy and macro-F1 by gate", INK, size=12, weight="600"))
    for v in (0.93, 0.94, 0.95, 0.96, 0.97):
        parts.append(line(px0, Ya(v), px0 + pw, Ya(v)))
        parts.append(text(px0 - 8, Ya(v) + 4, f"{v:.2f}", MUTED, anchor="end"))
    parts.append(line(px0, a0 + ah, px0 + pw, a0 + ah, BASE))
    for series, colour, name in ((acc, BLUE, "Accuracy"), (f1, ORANGE, "Macro-F1")):
        pts = " ".join(f"{X(g):.1f},{Ya(v):.1f}" for g, v in zip(gates, series))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        parts.append(text(px0 + pw + 10, Ya(series[-1]) + 4, name, SUB))
    sel = gates.index(0.09)
    parts.append(line(X(0.09), a0 - 6, X(0.09), b0 + bh, INK, 1))
    parts.append(text(X(0.09), a0 - 10, "selected gate: 9%", INK, anchor="middle", weight="600"))
    for series, colour, name in ((acc, BLUE, "accuracy"), (f1, ORANGE, "macro-F1")):
        parts.append(dot(X(0.09), Ya(series[sel]), colour, f"gate 9%: {name} {series[sel]:.4f}"))
        parts.append(text(X(0.09) + 10, Ya(series[sel]) + (-8 if name == "accuracy" else 16), f"{series[sel]:.3f}", INK, weight="600"))
    parts.append(legend([("Accuracy", BLUE, "line"), ("Macro-F1", ORANGE, "line")], px0 + 10, a0 + 18, 18))
    parts.append(text(px0, b0 - 16, "Instances sent to tracking (%)", INK, size=12, weight="600"))
    for v in (0, 10, 20, 30, 40):
        parts.append(line(px0, Yb(v), px0 + pw, Yb(v)))
        parts.append(text(px0 - 8, Yb(v) + 4, f"{v}", MUTED, anchor="end"))
    parts.append(line(px0, b0 + bh, px0 + pw, b0 + bh, BASE))
    pts = " ".join(f"{X(g):.1f},{Yb(v):.1f}" for g, v in zip(gates, trk))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{AQUA}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(dot(X(0.09), Yb(trk[sel]), AQUA, f"gate 9%: {trk[sel]:.1f}% of instances tracked"))
    parts.append(text(X(0.09) + 10, Yb(trk[sel]) - 8, f"{trk[sel]:.1f}%", INK, weight="600"))
    for g in (0.20, 0.15, 0.10, 0.05, 0.03):
        parts.append(line(X(g), b0 + bh, X(g), b0 + bh + 4, BASE))
        parts.append(text(X(g), b0 + bh + 18, f"{round(g * 100)}%", MUTED, anchor="middle"))
    parts.append(text(px0 + pw / 2, b0 + bh + 40, "Gate: track an instance when at least this share of its votes disagree (lower = more tracked)", SUB, anchor="middle"))
    parts.append("</svg>")
    return "".join(parts)


def figure_perclass() -> str:
    m = json.loads((REPORTS / "final_pipeline" / "metrics.json").read_text())
    before, after = m["before_tracking"]["per_class"], m["after_tracking"]["per_class"]
    order = sorted(after, key=lambda n: -after[n]["f1"])
    w, row = 700, 34
    h = 70 + row * len(order)
    px0, pw = 190, 470
    lo, hi = 0.70, 1.00

    def X(v: float) -> float:
        return px0 + pw * (v - lo) / (hi - lo)

    parts = [svg_open(w, h, "F1 for each instrument class before and after tracking")]
    parts.append(legend([("Before tracking", MUTED, "dot")], px0, 16, 16))
    parts.append(legend([("After tracking (9% gate)", BLUE, "dot")], px0 + 170, 16, 16))
    for v in (0.7, 0.8, 0.9, 1.0):
        parts.append(line(X(v), 34, X(v), 34 + row * len(order)))
        parts.append(text(X(v), 34 + row * len(order) + 16, f"{v:.1f}", MUTED, anchor="middle"))
    parts.append(text(px0 + pw / 2, 34 + row * len(order) + 38, "F1", SUB, anchor="middle"))
    weakest = {"Laparoscopic Grasper", "Clip Applier"}
    for i, name in enumerate(order):
        y = 34 + row * i + row / 2
        b, a = before[name]["f1"], after[name]["f1"]
        parts.append(text(px0 - 14, y + 4, name, INK, anchor="end"))
        parts.append(line(X(b), y, X(a), y, "#b7c4d6", 3))
        parts.append(dot(X(b), y, MUTED, f"{name}: {b:.3f} before tracking"))
        parts.append(dot(X(a), y, BLUE, f"{name}: {a:.3f} after tracking"))
        if name in weakest:
            parts.append(text(X(a) + 12, y + 4, f"{b:.3f} to {a:.3f}", INK, weight="600"))
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    out_dir = REPORTS / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("votes", figure_votes), ("roc", figure_roc), ("sweep", figure_sweep), ("perclass", figure_perclass)):
        path = out_dir / f"report_{name}.svg"
        path.write_text(fn(), encoding="utf-8")
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
