"""Builds the short PI report (docs/reports/classifier_report.html) from saved result files only.

Inputs under docs/reports/causal_realtime/:
    official_predictions.json   per-instance official-test predictions (scripts/export_official_predictions.py)
    latency_frames/*.json       frame-level latency runs (scripts/run_latency_matrix.sh)
Anything missing renders as "pending".

Usage:
    python scripts/build_pi_report.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
RT = REPO_ROOT / "docs" / "reports" / "causal_realtime"
OUT = REPO_ROOT / "docs" / "reports" / "classifier_report.html"

CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
SHORT = ["Bipolar", "Prograsp", "Needle Dr.", "Monopolar", "Suction", "Clip App.", "Grasper"]
INK, SUB = "#1e293b", "#475569"
FONT = 'Georgia, "Times New Roman", serif'
BUDGET_MS = 1000.0

CSS = """
:root{--navy:#1f3a5f;--ink:#1e293b;--sub:#475569;--panel:#f6f7f9;--border:#dfe3e8}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font-family:Georgia,"Times New Roman",serif;font-size:14px;line-height:1.4}
.wrap{max-width:820px;margin:0 auto;padding:28px 22px 40px}
h1{font-size:1.45rem;color:var(--navy);margin:0 0 2px}
.sub{color:var(--sub);font-size:.9rem;margin-bottom:.4em}
h2{font-size:1.02rem;color:var(--navy);border-bottom:1px solid var(--border);padding-bottom:3px;margin:1.1em 0 .4em}
table{border-collapse:collapse;width:100%;margin:.3em 0;font-size:.86rem}
th,td{border:1px solid var(--border);padding:3px 7px;text-align:left}
th{background:var(--panel)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th.grp{text-align:center}
.hi{background:#eaf3ee;font-weight:700}.off td{color:var(--sub)}.sep td{background:var(--panel);font-weight:700;font-size:.92em}
.ok{background:#eaf3ee}.warn{background:#fdf3e1}.bad{background:#fbe6e1;font-weight:700}
.pend{color:#898781;font-style:italic;border:1px dashed #c3c2b7;border-radius:3px;padding:0 4px;font-size:.8em}
.cap{color:var(--sub);font-size:.78rem;margin:.2em 0}
.cms{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-top:.5em}
svg{width:100%;height:auto;display:block}
a{color:#2a78d6}
@media print{@page{size:A4;margin:8mm}body{font-size:8.6pt;line-height:1.28}.wrap{max-width:none;padding:0}
h1{font-size:14pt}h2{font-size:10pt;margin:.7em 0 .3em}table{font-size:7.6pt}th,td{padding:1px 4px}
.cap{font-size:7.2pt}tr,svg{break-inside:avoid}h2{break-after:avoid}}
"""


def esc(s: str) -> str:
    return html.escape(s)


def txt(x: float, y: float, s: str, fill: str = INK, anchor: str = "start", size: float = 10, weight: str = "400") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{esc(s)}</text>'


def svg(w: int, h: int, label: str, body: str) -> str:
    return f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{esc(label)}" font-family=\'{FONT}\'>{body}</svg>'


def pipeline_figure() -> str:
    arrow = lambda d: f'<path d="{d}" stroke="{SUB}" stroke-width="1.4" fill="none" marker-end="url(#ah)"/>'

    def box(x, y, w, h, lines, fill="#f6f7f9", stroke="#dfe3e8"):
        out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/>'
        for i, s in enumerate(lines):
            out += txt(x + w / 2, y + 17 + i * 13, s, anchor="middle", size=10, weight="700" if i == 0 else "400")
        return out

    b = '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>'
    b += box(4, 66, 106, 56, ["Frame at time t", "+ ground-truth", "instrument mask"])
    b += arrow("M110,94 L124,94")
    b += box(126, 66, 92, 56, ["Crop", "mask-multiplied,", "made square"])
    b += arrow("M218,94 L232,94")
    b += box(234, 66, 124, 56, ["3-member classifier", "ResNet-50 320 px +", "2 MobileNetV3, 1 pass"])
    b += arrow("M358,94 L372,94")
    b += box(374, 66, 100, 56, ["Evidence", "Dirichlet from the", "logits; score S1"])
    b += arrow("M474,94 L486,94")
    b += f'<polygon points="488,94 548,56 608,94 548,132" fill="#fff" stroke="#1f3a5f"/>' + txt(548, 91, "Is S1 above", "#1f3a5f", "middle", 9.5, "700") + txt(548, 103, "the threshold?", "#1f3a5f", "middle", 9.5, "700")
    b += arrow("M608,94 L640,94") + txt(624, 87, "no", SUB, "middle", 9) + txt(624, 110, "87%", SUB, "middle", 8.5)
    b += box(642, 66, 134, 56, ["Output label", "from the single pass"], fill="#eaf3ee", stroke="#2f7d52")
    b += arrow("M548,132 L548,150 L474,150 L474,166") + txt(520, 146, "yes: uncertain (13%)", SUB, "end", 8.5)
    b += box(380, 168, 188, 62, ["Temporal tracker, past frames only", "EdgeTAM, 20 past frames", "or YOLO26s-seg, 15 past frames", "or SAM2-large, offline: 10 past + 10 future"])
    b += arrow("M568,199 L584,199")
    b += box(586, 168, 92, 62, ["Classify every", "tracked frame", "(same 3 members)"])
    b += arrow("M678,199 L692,199")
    b += box(694, 168, 82, 62, ["Evidence-", "weighted vote", "= output label"], fill="#eaf3ee", stroke="#2f7d52")
    label = "Pipeline from a frame and ground-truth mask through the classifier and an uncertainty gate to optional temporal tracking"
    return svg(780, 238, label, b).replace('viewBox="0 0 780 238"', 'viewBox="0 48 780 190"')


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion_matrix(y, pred, labels=range(7))
    n_total = int(cm.sum())
    per = []
    for c in range(7):
        tp = int(cm[c, c])
        n = int(cm[c].sum())
        fp = int(cm[:, c].sum()) - tp
        fn = n - tp
        tn = n_total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / n if n else 0.0
        per.append({"n": n, "acc": (tp + tn) / n_total, "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0})
    return {"cm": cm.tolist(), "per_class": per, "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=range(7)))}


def confusion_figure(cm: list[list[int]], title: str) -> str:
    w, h, x0, y0, cell = 296, 262, 66, 44, 31
    b = txt(4, 13, title, INK, size=11, weight="700") + txt(x0 + 3.5 * cell, 29, "predicted", SUB, "middle", 9.5, "700")
    b += f'<text transform="translate(10,{y0 + 3.5 * cell}) rotate(-90)" fill="{SUB}" text-anchor="middle" font-size="9.5" font-weight="700">true</text>'
    for j in range(7):
        b += txt(x0 + j * cell + cell / 2, y0 - 5, SHORT[j][:6], SUB, "middle", 8)
    for i in range(7):
        n = sum(cm[i])
        b += txt(x0 - 4, y0 + i * cell + cell / 2 + 3, SHORT[i], INK, "end", 9)
        for j in range(7):
            p = cm[i][j] / n if n else 0
            fill = f"rgba(42,120,214,{0.25 + 0.6 * p:.2f})" if i == j else f"rgba(235,104,52,{min(0.9, p * 6):.2f})"
            b += f'<rect x="{x0 + j * cell}" y="{y0 + i * cell}" width="{cell - 1}" height="{cell - 1}" fill="{fill}"><title>{esc(CLASSES[i])} predicted as {esc(CLASSES[j])}: {cm[i][j]} of {n} ({100 * p:.1f}%)</title></rect>'
            if p >= 0.005 or i == j:
                b += txt(x0 + j * cell + cell / 2, y0 + i * cell + cell / 2 + 3.5, f"{100 * p:.0f}" if p >= 0.095 or i == j else f"{100 * p:.1f}", INK, "middle", 8.5, "700" if i == j else "400")
    return svg(w, h, f"Confusion matrix, {title}, row-normalised percent", b)


def pend() -> str:
    return '<span class="pend">pending</span>'


def lat_cells(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text())["frame_ms"]
    lat = np.array([r[1] for r in rows])
    slow = max(rows, key=lambda r: r[1])
    return {"n": len(rows), "min": float(lat.min()), "mean": float(lat.mean()), "max": float(lat.max()),
            "slow_instruments": int(slow[0]), "instr_mean": float(np.mean([r[0] for r in rows])),
            "instr_max": int(max(r[0] for r in rows))}


def lat_cls(value: float, mean: float) -> str:
    return "bad" if mean >= BUDGET_MS else ("warn" if value >= BUDGET_MS else "ok")


def latency_table() -> str:
    names = [("SAM2-large", "sam2"), ("EdgeTAM", "edgetam"), ("YOLO26s-seg matching", "match")]
    head = ('<tr><th rowspan="2">tracker</th><th class="grp" colspan="3">typical frame (about 13% of instruments tracked)</th>'
            '<th class="grp" colspan="3">worst case (every instrument tracked)</th></tr>'
            '<tr><th class="n">fastest</th><th class="n">average</th><th class="n">slowest</th>'
            '<th class="n">fastest</th><th class="n">average</th><th class="n">slowest</th></tr>')
    rows, meta = "", None
    for label, key in names:
        typ = lat_cells(RT / "latency_frames" / f"{key}_share13.json")
        worst = lat_cells(RT / "latency_frames" / f"{key}_allgated.json")
        meta = meta or typ
        cells = f"<td>{label}</td>"
        for c in (typ, worst):
            if c is None:
                cells += f'<td class="n" colspan="3">{pend()}</td>'
            else:
                cells += (f'<td class="n {lat_cls(c["min"], c["min"])}">{c["min"]:.0f} ms</td>'
                          f'<td class="n {lat_cls(c["mean"], c["mean"])}">{c["mean"]:.0f} ms</td>'
                          f'<td class="n {lat_cls(c["max"], c["mean"])}">{c["max"]:.0f} ms</td>')
        rows += f"<tr>{cells}</tr>"
    budget = (f'<tr class="sep"><td colspan="7">Budget at 1 Hz: {BUDGET_MS:.0f} ms per frame. Green: within budget. '
              f'Amber: average within, slowest over. Red: average over.</td></tr>')
    info = f" Segment: {meta['n']} frames, {meta['instr_mean']:.1f} instruments per frame on average, up to {meta['instr_max']}." if meta else ""
    return (f'<table>{head}{budget}{rows}</table><p class="cap">Milliseconds to ingest one frame: detect, classify and track every flagged '
            f'instrument, all instruments of the frame as one unit. Titan Xp, nothing else running, one instrument tracked at a time. '
            f'The detector is a stand-in (YOLO26s-seg).{info}</p>')


def build() -> str:
    p = json.loads((RT / "official_predictions.json").read_text())
    y = np.array(p["y"])
    base3 = metrics(y, np.array(p["base3"]))
    a, b, s = (metrics(y, np.array(p[k]["pred"])) for k in ("A", "B", "S"))

    def row(label, m, tracked, cls=""):
        return (f'<tr class="{cls}"><td>{label}</td><td class="n">{tracked}</td><td class="n">{m["accuracy"]:.4f}</td>'
                f'<td class="n">{m["macro_f1"]:.4f}</td></tr>')

    results = (
        '<table><tr><th>configuration</th><th class="n">instances tracked</th><th class="n">accuracy</th><th class="n">macro-F1</th></tr>'
        + row("3-member ensemble, no tracking", base3, 0)
        + row("+ EdgeTAM (causal, 20 past frames)", a, sum(p["A"]["gated"]), "hi")
        + row("+ YOLO26s-seg (causal, 15 past frames)", b, sum(p["B"]["gated"]), "hi")
        + '<tr class="sep"><td colspan="4">Offline benchmark only: uses future frames, cannot run live</td></tr>'
        + row("4-member ensemble + SAM2-large (10 past + 10 future frames)", s, sum(p["S"]["gated"]), "off")
        + f'</table><p class="cap">Official test, 5 cases, {len(y):,} instances, one training seed.</p>')

    head = ('<tr><th rowspan="2">class</th><th class="n" rowspan="2">N</th><th class="grp" colspan="3">EdgeTAM</th>'
            '<th class="grp" colspan="3">YOLO26s-seg</th><th class="grp" colspan="3">SAM2-large (offline)</th></tr><tr>'
            + '<th class="n">acc</th><th class="n">recall</th><th class="n">F1</th>' * 3 + '</tr>')
    body = ""
    for c in range(7):
        body += f'<tr><td>{CLASSES[c]}</td><td class="n">{a["per_class"][c]["n"]}</td>'
        for m in (a, b, s):
            q = m["per_class"][c]
            body += f'<td class="n">{q["acc"]:.3f}</td><td class="n">{q["recall"]:.3f}</td><td class="n">{q["f1"]:.3f}</td>'
        body += "</tr>"
    perclass = f'<table>{head}{body}</table><p class="cap">acc: accuracy of that class against all others. recall: share of the class found.</p>'
    cms = '<div class="cms">' + confusion_figure(a["cm"], "EdgeTAM") + confusion_figure(b["cm"], "YOLO26s-seg") + confusion_figure(s["cm"], "SAM2-large (offline)") + "</div>"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Surgical Instrument Classifier Status</title><style>{CSS}</style></head><body><div class="wrap">
<h1>GraSP instrument classifier: how it works and where it stands</h1>
<div class="sub">Status 2026-10-01. The classifier names an instrument whose location is given by a ground-truth mask; detection is a separate component.</div>
<h2>How it works</h2>{pipeline_figure()}
<h2>Results</h2>{results}
<h2>Per class</h2>{perclass}{cms}
<h2>Error gallery</h2><p><a href="error_gallery.html">Every instance EdgeTAM, YOLO26s-seg or SAM2-large gets wrong, selectable by method</a></p>
<h2>Latency per frame</h2>{latency_table()}
</div></body></html>"""
    assert "—" not in page and "&mdash;" not in page, "no em dashes in the report"
    return page


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
