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
        "Closed jaws look like a plain suction tube. Checked against real crops, "
           "and a wide temporal search (about 13 s) found no frame with the jaws open.",
    ("Bipolar Forceps", "Prograsp Forceps"):
        "Look-alike pair: two independently trained models make the same mistake "
           "on the same crops.",
    ("Prograsp Forceps", "Bipolar Forceps"):
        "Look-alike pair: two independently trained models make the same mistake "
           "on the same crops.",
    ("Large Needle Driver", "Monopolar Curved Scissors"):
        "Look-alike pair: two independently trained models make the same mistake "
           "on the same crops.",
    ("Monopolar Curved Scissors", "Large Needle Driver"):
        "Look-alike pair: two independently trained models make the same mistake "
           "on the same crops.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gallery-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--total-instances", type=int, default=2861)
    parser.add_argument("--subtitle", default=(
        "Every misclassified official-test instance, current weighted ensemble "
        "(<code>configs/region_ensemble.yaml</code>, weight_resnet50_320=0.40), grouped by confusion pair. "
        "Each crop is exactly what the model was shown, not the raw frame."))
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
        badge = ""
        summary_rows.append(
            f'<tr><td><a href="#{anchor}">{html.escape(t)} &rarr; {html.escape(p)}</a>{badge}</td>'
            f'<td class="num">{len(items)}</td></tr>'
        )

    sections = []
    for (t, p), items in ordered_pairs:
        anchor = f"pair-{t.replace(' ', '_')}-{p.replace(' ', '_')}"
        cause = KNOWN_CAUSES.get((t, p))
        cause_html = f'<p class="cause">{html.escape(cause)}</p>' if cause else ""
        cards = []
        for m in items:
            src = img_data_uri(args.gallery_dir / m["image_path"])
            source = m.get("prediction_source", "single_frame")
            if "base_vote_share_pred" in m:
                votes = f'Votes before tracking: {m["base_vote_share_pred"]:.0%} predicted, {m["base_vote_share_true"]:.0%} true'
                if m.get("tracked"):
                    outcome = "broken by tracking" if m.get("was_correct_before") else "still wrong after tracking"
                    conf_html = f'<span class="conf tracked">{votes} &middot; {outcome}</span>'
                else:
                    conf_html = f'<span class="conf">{votes} &middot; not tracked</span>'
            elif source == "sam2_tracked":
                conf_html = '<span class="conf tracked">SAM2 temporal-track prediction (confidence-gated, docs/DECISIONS.md 2026-09-16)</span>'
            else:
                conf_html = f'<span class="conf">P(pred)={m["pred_confidence"]:.2f} &middot; P(true)={m["true_confidence"]:.2f}</span>'
            cards.append(f'''
        <figure class="card">
          <img src="{src}" alt="{html.escape(t)} misclassified as {html.escape(p)}" loading="lazy">
          <figcaption>
            <span class="tag true">true: {html.escape(t)}</span>
            <span class="tag pred">predicted: {html.escape(p)}</span>
            {conf_html}
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
<title>Error Gallery</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --navy: #1f3a5f;
  --ink: #1e293b;
  --sub: #475569;
  --bg: #ffffff;
  --panel: #f6f7f9;
  --border: #dfe3e8;
  --link: #5b7ea8;
  --good: #2f7d52;
  --good-soft: #eaf3ee;
  --bad: #a8452d;
  --bad-soft: #fdf3ee;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-size: 15.5px; line-height: 1.55; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px 80px; }}
h1 {{ font-size: 1.6rem; color: var(--navy); margin: 0 0 6px; }}
.subtitle {{ color: var(--sub); font-size: 0.95rem; margin-bottom: 1.4em; max-width: 780px; }}
.stat-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 1.2em 0 1.8em; }}
.stat {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; min-width: 130px; }}
.stat .n {{ font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; display: block; color: var(--navy); }}
.stat .label {{ color: var(--sub); font-size: 0.85rem; }}
h2 {{ font-size: 1.1rem; color: var(--navy); margin: 0 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--border); display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }}
h2 .arrow {{ color: var(--sub); font-weight: 400; }}
h2 .count {{ margin-left: auto; color: var(--sub); font-size: 0.85rem; font-weight: 400; }}
p.cause {{ font-size: 0.9rem; color: var(--sub); margin: 0.5em 0 1em; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 0.92rem; }}
th, td {{ border: 1px solid var(--border); padding: 5px 9px; text-align: left; }}
th {{ background: var(--panel); font-weight: 700; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.tag {{ display: inline-block; font-size: 0.75rem; padding: 1px 7px; border-radius: 4px; }}
.tag.true {{ background: var(--good-soft); color: var(--good); }}
.tag.pred {{ background: var(--bad-soft); color: var(--bad); }}
.tablewrap {{ overflow-x: auto; margin-bottom: 2.2em; }}
.pair-section {{ margin-bottom: 2.4em; }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }}
.card {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px; margin: 0; overflow: hidden; display: flex; flex-direction: column; }}
.card img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #0b0b0b; display: block; }}
.card figcaption {{ padding: 7px 9px 9px; display: flex; flex-direction: column; gap: 3px; }}
.card .conf {{ font-size: 0.75rem; color: var(--sub); }}
.card .conf.tracked {{ color: var(--good); }}
.card .src {{ font-size: 0.7rem; color: var(--sub); opacity: 0.8; overflow-wrap: anywhere; }}
a {{ color: var(--link); }}
</style>

<div class="wrap">
  <h1>Error Gallery</h1>
  <div class="subtitle">{args.subtitle}</div>

  <div class="stat-row">
    <div class="stat"><span class="n">{args.total_instances - n_wrong} / {args.total_instances}</span><span class="label">correct</span></div>
    <div class="stat"><span class="n">{accuracy:.1%}</span><span class="label">accuracy</span></div>
    <div class="stat"><span class="n">{n_wrong}</span><span class="label">errors</span></div>
    <div class="stat"><span class="n">{len(ordered_pairs)}</span><span class="label">confusion pairs</span></div>
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
