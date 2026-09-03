"""Renders the crops + manifest.json from generate_error_gallery.py into a
single self-contained HTML page (images embedded as data URIs) -- a
browsable, no-download-needed view of every misclassified official-test
instance, grouped by confusion pair.

Usage:
    python scripts/build_error_gallery_page.py \\
        --gallery-dir error_gallery --out error_gallery_page.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Confusion pairs with a documented, checked root cause (docs/DECISIONS.md).
# Anything not listed here is shown without a cause badge -- not guessed.
KNOWN_CAUSES = {
    ("Laparoscopic Grasper", "Suction Instrument"):
        "Confirmed physical-state ceiling: when this instrument's jaws are closed it is a smooth, "
        "featureless shaft with no visual information distinguishing it from a suction tube. "
        "Checked directly against real crops; wide-window temporal search (~13s) found no frame "
        "where the jaws open. Not a training-objective gap.",
    ("Bipolar Forceps", "Prograsp Forceps"):
        "Genuinely similar instrument pair -- confirmed because two independently-trained models "
        "make the identical mistake on the identical crops, not one model's quirk.",
    ("Prograsp Forceps", "Bipolar Forceps"):
        "Genuinely similar instrument pair -- confirmed because two independently-trained models "
        "make the identical mistake on the identical crops, not one model's quirk.",
    ("Large Needle Driver", "Monopolar Curved Scissors"):
        "Genuinely similar instrument pair -- confirmed because two independently-trained models "
        "make the identical mistake on the identical crops, not one model's quirk.",
    ("Monopolar Curved Scissors", "Large Needle Driver"):
        "Genuinely similar instrument pair -- confirmed because two independently-trained models "
        "make the identical mistake on the identical crops, not one model's quirk.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-instances", type=int, default=2861)
    return parser.parse_args()


def img_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.gallery_dir / "manifest.json").read_text())

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in manifest:
        groups[(m["true_class"], m["pred_class"])].append(m)
    ordered_pairs = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    n_wrong = len(manifest)
    accuracy = 1 - n_wrong / args.total_instances

    summary_rows = []
    for (t, p), items in ordered_pairs:
        anchor = f"pair-{t.replace(' ', '_')}-{p.replace(' ', '_')}"
        badge = ' <span class="tag known">known cause</span>' if (t, p) in KNOWN_CAUSES else ""
        summary_rows.append(
            f'<tr><td><a href="#{anchor}">{html.escape(t)} &rarr; {html.escape(p)}</a>{badge}</td>'
            f'<td class="num">{len(items)}</td></tr>'
        )

    sections = []
    for (t, p), items in ordered_pairs:
        anchor = f"pair-{t.replace(' ', '_')}-{p.replace(' ', '_')}"
        cause = KNOWN_CAUSES.get((t, p))
        cause_html = f'<p class="cause">{html.escape(cause)}</p>' if cause else (
            '<p class="cause unknown">Not yet individually diagnosed.</p>'
        )
        cards = []
        for m in items:
            src = img_data_uri(args.gallery_dir / m["image_path"])
            cards.append(f'''
        <figure class="card">
          <img src="{src}" alt="{html.escape(t)} misclassified as {html.escape(p)}" loading="lazy">
          <figcaption>
            <span class="tag true">true: {html.escape(t)}</span>
            <span class="tag pred">pred: {html.escape(p)}</span>
            <span class="conf">P(pred)={m['pred_confidence']:.2f} &middot; P(true)={m['true_confidence']:.2f}</span>
            <span class="src">{html.escape(m['file_name'])}</span>
          </figcaption>
        </figure>''')
        sections.append(f'''
    <section class="pair-section" id="{anchor}">
      <h2><span class="from">{html.escape(t)}</span><span class="arrow">&rarr;</span><span class="to">{html.escape(p)}</span>
        <span class="count">{len(items)} instance{"s" if len(items) != 1 else ""}</span></h2>
      {cause_html}
      <div class="card-grid">{"".join(cards)}</div>
    </section>''')

    page = f'''<!doctype html>
<title>Task B Error Gallery</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --ink: #16212b;
  --sub: #56666f;
  --bg: #f6f7f6;
  --panel: #ffffff;
  --border: #dbe1e0;
  --accent: #2f6f6a;
  --accent-soft: #e3efed;
  --true: #2f6f6a;
  --pred: #a8452d;
  --pred-soft: #f7e9e4;
  --known: #92721c;
  --known-soft: #f4ecd6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ink: #e7ece9;
    --sub: #9db0a9;
    --bg: #10171a;
    --panel: #16201f;
    --border: #2a3634;
    --accent: #6cbdb2;
    --accent-soft: #1c2e2c;
    --true: #6cbdb2;
    --pred: #e08464;
    --pred-soft: #2e211b;
    --known: #d8b45c;
    --known-soft: #2c2510;
  }}
}}
:root[data-theme="dark"] {{
  --ink: #e7ece9;
  --sub: #9db0a9;
  --bg: #10171a;
  --panel: #16201f;
  --border: #2a3634;
  --accent: #6cbdb2;
  --accent-soft: #1c2e2c;
  --true: #6cbdb2;
  --pred: #e08464;
  --pred-soft: #2e211b;
  --known: #d8b45c;
  --known-soft: #2c2510;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: "IBM Plex Sans", Arial, sans-serif;
  font-size: 15px; line-height: 1.5;
}}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }}
h1 {{ font-size: 1.7rem; margin: 0 0 6px; text-wrap: balance; letter-spacing: -0.01em; }}
.subtitle {{ color: var(--sub); font-size: 0.95rem; margin-bottom: 1.6em; }}
.subtitle code {{ font-family: "IBM Plex Mono", monospace; background: var(--panel); border: 1px solid var(--border); padding: 1px 5px; border-radius: 4px; }}
.stat-row {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 1.4em 0 2em; }}
.stat {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 18px; min-width: 140px;
}}
.stat .n {{ font-family: "IBM Plex Mono", monospace; font-size: 1.5rem; font-weight: 600; font-variant-numeric: tabular-nums; display: block; }}
.stat .label {{ color: var(--sub); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em; }}
h2 {{
  font-size: 1.05rem; margin: 0 0 10px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; font-family: "IBM Plex Mono", monospace;
}}
h2 .from {{ color: var(--true); font-weight: 600; }}
h2 .arrow {{ color: var(--sub); }}
h2 .to {{ color: var(--pred); font-weight: 600; }}
h2 .count {{ margin-left: auto; color: var(--sub); font-size: 0.85rem; font-family: "IBM Plex Sans", sans-serif; font-weight: 400; }}
p.cause {{ font-size: 0.88rem; color: var(--ink); background: var(--known-soft); border-left: 3px solid var(--known); padding: 8px 12px; border-radius: 0 6px 6px 0; margin: 0.6em 0 1.1em; }}
p.cause.unknown {{ background: transparent; border-left-color: var(--border); color: var(--sub); font-style: italic; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 0.9rem; }}
th, td {{ border-bottom: 1px solid var(--border); padding: 7px 10px; text-align: left; }}
th {{ color: var(--sub); font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.04em; }}
td.num, th.num {{ text-align: right; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }}
.tag {{ display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; padding: 2px 7px; border-radius: 5px; font-weight: 500; }}
.tag.known {{ background: var(--known-soft); color: var(--known); margin-left: 6px; }}
.tag.true {{ background: var(--accent-soft); color: var(--true); }}
.tag.pred {{ background: var(--pred-soft); color: var(--pred); }}
.tablewrap {{ overflow-x: auto; margin-bottom: 2.4em; }}
.pair-section {{ margin-bottom: 2.6em; }}
.card-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px;
}}
.card {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  margin: 0; overflow: hidden; display: flex; flex-direction: column;
}}
.card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #0b0b0b; display: block; }}
.card figcaption {{ padding: 8px 9px 10px; display: flex; flex-direction: column; gap: 4px; }}
.card .conf {{ font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: var(--sub); }}
.card .src {{ font-family: "IBM Plex Mono", monospace; font-size: 0.65rem; color: var(--sub); opacity: 0.75; overflow-wrap: anywhere; }}
a {{ color: var(--accent); }}
</style>

<div class="wrap">
  <h1>Task B Error Gallery</h1>
  <div class="subtitle">Every misclassified official-test instance, current weighted ensemble
    (<code>configs/region_ensemble.yaml</code>, weight_resnet50_320=0.40), grouped by confusion pair.
    Each crop is exactly what the model was shown, not the raw frame.</div>

  <div class="stat-row">
    <div class="stat"><span class="n">{args.total_instances - n_wrong} / {args.total_instances}</span><span class="label">correct</span></div>
    <div class="stat"><span class="n">{accuracy:.1%}</span><span class="label">accuracy</span></div>
    <div class="stat"><span class="n">{n_wrong}</span><span class="label">total errors</span></div>
    <div class="stat"><span class="n">{len(ordered_pairs)}</span><span class="label">distinct confusion pairs</span></div>
  </div>

  <div class="tablewrap">
    <table>
      <tr><th>confusion pair</th><th class="num">count</th></tr>
      {"".join(summary_rows)}
    </table>
  </div>

  {"".join(sections)}
</div>
'''

    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {n_wrong} images embedded)")


if __name__ == "__main__":
    main()
