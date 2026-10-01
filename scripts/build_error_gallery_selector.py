"""Builds docs/reports/error_gallery.html: official-test errors of EdgeTAM (A), YOLO26s-seg (B) and SAM2-large
offline (S), one method at a time, chosen from a selector at the top (CSS only, no scripts).

Inputs: docs/reports/gallery_assets/data.json and crops/ (scripts/build_error_gallery_data.py) and
docs/reports/causal_realtime/official_predictions.json.

Usage:
    python scripts/build_error_gallery_selector.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "docs" / "reports"
CLASSES = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
           "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
METHODS = [("A", "EdgeTAM (causal)", "mA"), ("B", "YOLO26s-seg (causal)", "mB"), ("S", "SAM2-large (offline)", "mS")]

CSS = """
:root{--navy:#1f3a5f;--ink:#1e293b;--sub:#475569;--panel:#f6f7f9;--border:#dfe3e8}
*{box-sizing:border-box}
body{margin:0;font-family:Georgia,"Times New Roman",serif;color:var(--ink);font-size:14px}
.wrap{max-width:1000px;margin:0 auto;padding:24px 20px 50px}
h1{font-size:1.4rem;color:var(--navy);margin:0 0 4px}.sub{color:var(--sub);font-size:.88rem;margin-bottom:12px}
input[type=radio]{position:absolute;opacity:0;pointer-events:none}
.bar{display:flex;gap:8px;position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid var(--border);z-index:2}
.bar label{padding:6px 14px;border:1px solid var(--border);border-radius:6px;background:var(--panel);cursor:pointer;font-weight:700;font-size:.88rem}
.sections section{display:none}
#mA:checked ~ .bar label[for=mA],#mB:checked ~ .bar label[for=mB],#mS:checked ~ .bar label[for=mS]{background:var(--navy);color:#fff;border-color:var(--navy)}
#mA:checked ~ .sections #sA,#mB:checked ~ .sections #sB,#mS:checked ~ .sections #sS{display:block}
.pair{margin:18px 0 6px;font-weight:700;color:var(--navy);border-bottom:1px solid var(--border);padding-bottom:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(178px,1fr));gap:10px}
.card{border:1px solid var(--border);border-radius:6px;padding:6px;background:#fff;font-size:.78rem;line-height:1.35}
.card img{width:100%;height:auto;display:block;border-radius:4px;background:#000}
.tag{display:inline-block;border-radius:3px;padding:0 5px;font-size:.72rem;font-weight:700}
.broken{background:#fbe6e1;color:#8a2a12}.unfixed{background:#fdf3e1;color:#7a5200}.untracked{background:var(--panel);color:var(--sub)}
.small{color:var(--sub)}
"""


def build() -> str:
    data = json.loads((REPORTS / "gallery_assets" / "data.json").read_text())
    pred = json.loads((REPORTS / "causal_realtime" / "official_predictions.json").read_text())
    base_of = {"A": pred["base3"], "B": pred["base3"], "S": pred["S"]["base"]}
    y = pred["y"]
    inputs = "".join(f'<input type="radio" name="m" id="{rid}"{" checked" if i == 0 else ""}>' for i, (_, _, rid) in enumerate(METHODS))
    bar_items, sections = "", ""
    for key, label, rid in METHODS:
        wrong = [r for r in data if not r[key]["correct"]]
        bar_items += f'<label for="{rid}">{html.escape(label)}: {len(wrong)} errors</label>'
        groups: dict[tuple[str, str], list[dict]] = {}
        for r in sorted(wrong, key=lambda r: (r["true"], r[key]["pred"], r["index"])):
            groups.setdefault((r["true"], r[key]["pred"]), []).append(r)
        body = ""
        for (true, wrong_pred), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            cards = ""
            for r in items:
                base = CLASSES[base_of[key][r["index"]]]
                base_right = base_of[key][r["index"]] == y[r["index"]]
                gated = r[key]["gated"]
                tag = ('<span class="tag broken">tracking broke it</span>' if gated and base_right
                       else '<span class="tag unfixed">tracked, not fixed</span>' if gated
                       else '<span class="tag untracked">not tracked</span>')
                score = f' | S1 {r["s1"]:.1e}' if key != "S" else ""
                cards += (f'<div class="card"><img loading="lazy" src="gallery_assets/crops/{r["index"]}.jpg" alt="">'
                          f'<div><b>{html.escape(true)}</b> read as <b>{html.escape(wrong_pred)}</b></div>'
                          f'<div class="small">single pass: {html.escape(base)}{score}</div>{tag}'
                          f'<div class="small">{html.escape(r["file"])}</div></div>')
            body += f'<div class="pair">{html.escape(true)} read as {html.escape(wrong_pred)} ({len(items)})</div><div class="grid">{cards}</div>'
        sections += f'<section id="s{key}">{body}</section>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Error Gallery</title><style>{CSS}</style></head><body><div class="wrap"><h1>Error gallery, official test</h1>'
            f'<div class="sub">Every instance the selected method gets wrong. Each picture is the mask-multiplied crop the classifier sees. '
            f'Grouped by true class and wrong prediction, largest groups first. <a href="classifier_report.html">Back to the report</a></div>'
            f'{inputs}<div class="bar">{bar_items}</div><div class="sections">{sections}</div></div></body></html>')


def main() -> None:
    page = build()
    assert "—" not in page
    out = REPORTS / "error_gallery.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out, f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
