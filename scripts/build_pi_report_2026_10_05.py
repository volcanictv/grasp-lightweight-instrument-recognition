"""Builds the PI report page. Every number below comes from a result file or a logged result (see the source line at the bottom of the page)."""
from pathlib import Path

OUT = Path(r"C:\Users\aryan\.claude\jobs\02cd3b07\tmp\pi_report.html")

# ---------- chart 1: headline dot plot, shared axis ----------
AX_MIN, AX_MAX = 74.0, 90.0
W, LEFT, RIGHT = 760, 150, 30
PW = W - LEFT - RIGHT


def x(v):
    return LEFT + (v - AX_MIN) / (AX_MAX - AX_MIN) * PW


rows = [
    ("mIoU", 87.37, (86.85, 88.21), 86.61, 86.36),
    ("IoU", 86.22, (85.56, 87.09), 83.38, 83.51),
    ("mcIoU", 78.33, (76.33, 79.81), 77.42, 77.54),
]
RH, TOP = 78, 36
H = TOP + RH * len(rows) + 46
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Dot plot of mIoU, IoU and mcIoU for this pipeline and TAPIS" class="chart">']
for t in range(int(AX_MIN), int(AX_MAX) + 1, 2):
    svg.append(f'<line x1="{x(t):.1f}" y1="{TOP - 14}" x2="{x(t):.1f}" y2="{H - 40}" class="grid"/>')
    svg.append(f'<text x="{x(t):.1f}" y="{H - 22}" class="tick" text-anchor="middle">{t}</text>')
for i, (name, ours, ci, tapis, vst) in enumerate(rows):
    cy = TOP + RH * i + RH / 2
    svg.append(f'<text x="{LEFT - 16}" y="{cy + 5:.1f}" class="rowlab" text-anchor="end">{name}</text>')
    svg.append(f'<line x1="{x(ci[0]):.1f}" y1="{cy}" x2="{x(ci[1]):.1f}" y2="{cy}" class="ci"/>')
    for cx_ in (ci[0], ci[1]):
        svg.append(f'<line x1="{x(cx_):.1f}" y1="{cy - 7}" x2="{x(cx_):.1f}" y2="{cy + 7}" class="ci"/>')
    svg.append(f'<circle cx="{x(ours):.1f}" cy="{cy}" r="7" class="dot-ours"/>')
    svg.append(f'<text x="{x(ours):.1f}" y="{cy - 14}" class="val ours" text-anchor="middle">{ours:.2f}</text>')
    dy = 16 if name != "mcIoU" else 16
    svg.append(f'<rect x="{x(tapis) - 6:.1f}" y="{cy - 6}" width="12" height="12" class="dot-tapis"/>')
    svg.append(f'<path d="M{x(vst):.1f} {cy - 7} L{x(vst) + 7:.1f} {cy + 6} L{x(vst) - 7:.1f} {cy + 6} Z" class="dot-vst"/>')
    svg.append(f'<text x="{x((tapis + vst) / 2):.1f}" y="{cy + 29:.1f}" class="val" text-anchor="middle">{tapis:.2f} / {vst:.2f}</text>')
svg.append(f'<text x="{LEFT + PW / 2}" y="{H - 4}" class="axislab" text-anchor="middle">score (axis starts at 74)</text>')
svg.append("</svg>")
CHART1 = "\n".join(svg)

# ---------- chart 2: per-class IoU bars ----------
classes = [
    ("Monopolar Curved Scissors", 94.2),
    ("Large Needle Driver", 87.0),
    ("Bipolar Forceps", 84.0),
    ("Suction Instrument", 78.1),
    ("Clip Applier", 76.8),
    ("Prograsp Forceps", 67.5),
    ("Laparoscopic Grasper", 60.7),
]
CW, CL, CR = 760, 210, 50
CPW = CW - CL - CR
BH = 28
CH = 14 + len(classes) * BH + 34
s2 = [f'<svg viewBox="0 0 {CW} {CH}" role="img" aria-label="Per-class IoU bars" class="chart">']
for t in range(0, 101, 20):
    gx = CL + t / 100 * CPW
    s2.append(f'<line x1="{gx:.1f}" y1="6" x2="{gx:.1f}" y2="{CH - 30}" class="grid"/>')
    s2.append(f'<text x="{gx:.1f}" y="{CH - 12}" class="tick" text-anchor="middle">{t}</text>')
for i, (n, v) in enumerate(classes):
    y0 = 14 + i * BH
    cls = "bar weak" if v < 70 else "bar"
    s2.append(f'<text x="{CL - 10}" y="{y0 + 15}" class="rowlab" text-anchor="end">{n}</text>')
    s2.append(f'<rect x="{CL}" y="{y0 + 3}" width="{v / 100 * CPW:.1f}" height="18" class="{cls}"/>')
    s2.append(f'<text x="{CL + v / 100 * CPW + 6:.1f}" y="{y0 + 16}" class="val">{v:.1f}</text>')
s2.append("</svg>")
CHART2 = "\n".join(s2)

# ---------- chart 3: tracking budget ladder (mIoU) ----------
lad = [
    ("No tracking", 83.47),
    ("Track 539 (19%)", 86.52),
    ("Track 833 (29%)", 87.37),
    ("Held-out gate (42 to 47%)", 87.50),
]
TW, TL, TR = 760, 210, 50
TPW = TW - TL - TR
TMIN, TMAX = 80.0, 92.0
TBH = 28
TH = 14 + (len(lad) + 1) * TBH + 34


def tx(v):
    return TL + (v - TMIN) / (TMAX - TMIN) * TPW


s3 = [f'<svg viewBox="0 0 {TW} {TH}" role="img" aria-label="mIoU by tracking budget" class="chart">']
for t in range(80, 93, 2):
    s3.append(f'<line x1="{tx(t):.1f}" y1="6" x2="{tx(t):.1f}" y2="{TH - 30}" class="grid"/>')
    s3.append(f'<text x="{tx(t):.1f}" y="{TH - 12}" class="tick" text-anchor="middle">{t}</text>')
for i, (n, v) in enumerate(lad):
    y0 = 14 + i * TBH
    s3.append(f'<text x="{TL - 10}" y="{y0 + 15}" class="rowlab" text-anchor="end">{n}</text>')
    s3.append(f'<rect x="{TL}" y="{y0 + 3}" width="{tx(v) - TL:.1f}" height="18" class="bar"/>')
    s3.append(f'<text x="{tx(v) + 6:.1f}" y="{y0 + 16}" class="val">{v:.2f}</text>')
y0 = 14 + len(lad) * TBH
s3.append(f'<text x="{TL - 10}" y="{y0 + 15}" class="rowlab" text-anchor="end">Perfect classes (ceiling)</text>')
s3.append(f'<rect x="{TL}" y="{y0 + 3}" width="{tx(91.17) - TL:.1f}" height="18" class="bar ceil"/>')
s3.append(f'<text x="{tx(91.17) + 6:.1f}" y="{y0 + 16}" class="val">91.17</text>')
s3.append(f'<line x1="{tx(86.61):.1f}" y1="6" x2="{tx(86.61):.1f}" y2="{TH - 30}" class="ref"/>')
s3.append(f'<text x="{tx(86.61):.1f}" y="{TH - 1}" class="tick ref-lab" text-anchor="middle">TAPIS 86.61</text>')
s3.append("</svg>")
CHART3 = "\n".join(s3)

# ---------- chart 4: fold1 gate sweep (accuracy vs share tracked) ----------
sweep = [(2.0, .9379), (4.0, .9431), (6.0, .9471), (8.0, .9527), (10.0, .9546), (12.0, .9561), (15.0, .9589), (20.0, .9604), (25.0, .9604), (31.2, .9607)]
BASE = .9357
GW, GH = 760, 260
GL, GR, GT, GB = 64, 24, 20, 44
YMIN, YMAX = .93, .965
XMAX = 32.0


def gx(v):
    return GL + v / XMAX * (GW - GL - GR)


def gy(v):
    return GT + (YMAX - v) / (YMAX - YMIN) * (GH - GT - GB)


s4 = [f'<svg viewBox="0 0 {GW} {GH}" role="img" aria-label="Fold1 accuracy against share of instruments tracked" class="chart">']
for t in (0.93, 0.94, 0.95, 0.96):
    s4.append(f'<line x1="{GL}" y1="{gy(t):.1f}" x2="{GW - GR}" y2="{gy(t):.1f}" class="grid"/>')
    s4.append(f'<text x="{GL - 8}" y="{gy(t) + 4:.1f}" class="tick" text-anchor="end">{t:.2f}</text>')
for t in (0, 5, 10, 15, 20, 25, 30):
    s4.append(f'<text x="{gx(t):.1f}" y="{GH - 24}" class="tick" text-anchor="middle">{t}%</text>')
s4.append(f'<line x1="{GL}" y1="{gy(BASE):.1f}" x2="{GW - GR}" y2="{gy(BASE):.1f}" class="ref"/>')
s4.append(f'<text x="{GW - GR}" y="{gy(BASE) + 15:.1f}" class="tick ref-lab" text-anchor="end">no tracking 0.9357</text>')
pts = " ".join(f"{gx(a):.1f},{gy(b):.1f}" for a, b in sweep)
s4.append(f'<polyline points="{pts}" fill="none" class="ci"/>')
for a, b in sweep:
    s4.append(f'<circle cx="{gx(a):.1f}" cy="{gy(b):.1f}" r="4" class="dot-ours"/>')
s4.append(f'<circle cx="{gx(20.0):.1f}" cy="{gy(.9604):.1f}" r="9" fill="none" class="ci"/>')
s4.append(f'<text x="{gx(20.0):.1f}" y="{gy(.9604) - 16:.1f}" class="val ours" text-anchor="middle">chosen: 20%, 0.9604</text>')
s4.append(f'<text x="{GL + (GW - GL - GR) / 2:.1f}" y="{GH - 4}" class="axislab" text-anchor="middle">share of instruments tracked (held-out fold1)</text>')
s4.append("</svg>")
CHART4 = "\n".join(s4)

HTML = """<title>GraSP Instrument Recognition</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one narrow reading column, a table or chart per section, little prose. */
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --fg: #1c2430;
  --muted: #5d6877;
  --rule: #d9dee5;
  --accent: #0b6e8a;
  --accent-soft: #dcedf2;
  --tapis: #8a94a3;
  --warn: #9a5b00;
  --warn-soft: #fbefd9;
  --good: #1d7a46;
  --good-soft: #dff1e6;
  --bad: #a3322b;
  --bad-soft: #f7e0de;
  --font: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12171e; --surface: #1a212b; --fg: #e4e9f0; --muted: #98a3b3; --rule: #2d3745;
    --accent: #4fb3d1; --accent-soft: #17323c; --tapis: #7c8797;
    --warn: #e2a64d; --warn-soft: #3a2b12; --good: #5cc589; --good-soft: #143122; --bad: #ee8780; --bad-soft: #3b1b19;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #12171e; --surface: #1a212b; --fg: #e4e9f0; --muted: #98a3b3; --rule: #2d3745;
  --accent: #4fb3d1; --accent-soft: #17323c; --tapis: #7c8797;
  --warn: #e2a64d; --warn-soft: #3a2b12; --good: #5cc589; --good-soft: #143122; --bad: #ee8780; --bad-soft: #3b1b19;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font-family: var(--font); font-size: 15px; line-height: 1.5; margin: 0; padding-inline: 20px; padding-block: 36px 64px; }
main { max-width: 860px; margin-inline: auto; display: flex; flex-direction: column; gap: 44px; }
h1 { font-size: 30px; line-height: 1.15; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px; text-wrap: balance; }
h2 { font-size: 13px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 0 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--rule); }
p { margin: 0 0 10px; }
.sub { color: var(--muted); margin: 0; }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 18px; }
.fact { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 12px; min-width: 0; }
.fact b { display: block; font-family: var(--mono); font-size: 19px; font-weight: 500; font-variant-numeric: tabular-nums; }
.fact span { color: var(--muted); font-size: 12.5px; }
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
tr:last-child td { border-bottom: 0; }
th { font-size: 12px; font-weight: 600; color: var(--muted); letter-spacing: 0.03em; background: var(--bg); white-space: nowrap; }
td.n, th.n { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.ours td { background: var(--accent-soft); font-weight: 500; }
tr.sub td { color: var(--muted); }
.chip { display: inline-block; font-size: 11.5px; font-weight: 600; padding: 1px 8px; border-radius: 99px; white-space: nowrap; }
.chip.good { background: var(--good-soft); color: var(--good); }
.chip.bad { background: var(--bad-soft); color: var(--bad); }
.chip.warn { background: var(--warn-soft); color: var(--warn); }
.chip.neutral { background: var(--bg); color: var(--muted); border: 1px solid var(--rule); }
.note { font-size: 13px; color: var(--muted); margin: 8px 0 0; }
.callout { background: var(--warn-soft); color: var(--fg); border-left: 3px solid var(--warn); border-radius: 4px; padding: 10px 14px; font-size: 14px; margin-top: 12px; }
.chartbox { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 12px 8px 4px; overflow-x: auto; }
.chart { width: 100%; min-width: 560px; height: auto; display: block; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .tick, .chart .axislab { fill: var(--muted); font-family: var(--mono); font-size: 11px; }
.chart .rowlab { fill: var(--fg); font-family: var(--font); font-size: 13px; }
.chart .val { fill: var(--muted); font-family: var(--mono); font-size: 12px; }
.chart .val.ours { fill: var(--accent); font-weight: 500; }
.chart .ci { stroke: var(--accent); stroke-width: 2; }
.chart .dot-ours { fill: var(--accent); }
.chart .dot-tapis { fill: var(--tapis); }
.chart .dot-vst { fill: none; stroke: var(--tapis); stroke-width: 2; }
.chart .bar { fill: var(--accent); }
.chart .bar.weak { fill: var(--bad); }
.chart .bar.ceil { fill: var(--tapis); opacity: 0.55; }
.chart .ref { stroke: var(--tapis); stroke-width: 1.5; stroke-dasharray: 4 3; }
.chart .ref-lab { fill: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px; color: var(--muted); padding: 4px 8px 10px; }
.legend i { display: inline-block; width: 11px; height: 11px; margin-right: 6px; vertical-align: -1px; }
.flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; counter-reset: s; }
.step { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 11px; font-size: 13px; min-width: 0; }
.step b { display: block; font-size: 13.5px; margin-bottom: 3px; }
.step span { color: var(--muted); }
.step.gate { border-color: var(--accent); background: var(--accent-soft); }
@media (max-width: 720px) { .flow { grid-template-columns: 1fr; } }
footer { color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 14px; }
code { font-family: var(--mono); font-size: 12.5px; }
</style>

<main>

<header>
  <h1>GraSP Instrument Recognition</h1>
  <p class="sub">Project overview for the PI. Results as of 2026-10-05, official GraSP test set.</p>
  <div class="facts">
    <div class="fact"><b>87.37</b><span>mIoU, TAPIS 86.61</span></div>
    <div class="fact"><b>86.22</b><span>IoU, TAPIS 83.38</span></div>
    <div class="fact"><b>78.33</b><span>mcIoU, TAPIS 77.42</span></div>
    <div class="fact"><b>0.957</b><span>instrument accuracy</span></div>
  </div>
  <div class="callout">Our pipeline is given the ground-truth boxes. TAPIS finds its own. The comparison is an upper bound for a pipeline fed by a detector; the IoU margin is the most inflated by this.</div>
</header>

<section>
  <h2>Task</h2>
  <div class="tablewrap"><table>
    <tr><th>Item</th><th>Value</th></tr>
    <tr><td>Dataset</td><td>GraSP, robotic prostatectomy video, 7 instrument classes</td></tr>
    <tr><td>Official test set</td><td>2,861 instruments in 1,125 frames, 5 cases (about 1 frame per second)</td></tr>
    <tr><td>Input to our pipeline</td><td>Frames plus one box per instrument (ground truth for now)</td></tr>
    <tr><td>Output</td><td>Instance mask, class, an uncertainty score per instrument</td></tr>
    <tr><td>Scores</td><td>mIoU, IoU, mcIoU from the benchmark's own code (ported from MATIS)</td></tr>
    <tr><td>Compute</td><td>2 x Titan Xp, 4 CPU cores</td></tr>
  </table></div>
</section>

<section>
  <h2>Pipeline</h2>
  <div class="flow">
    <div class="step"><b>1. Box</b><span>ground truth, later a detector</span></div>
    <div class="step"><b>2. Segment</b><span>SAM2.1-large + SAM3 fine-tuned on GraSP, mirror-averaged</span></div>
    <div class="step"><b>3. Classify</b><span>4-member evidential ensemble, one pass, gives class and uncertainty</span></div>
    <div class="step gate"><b>4. Gate</b><span>uncertain instruments only (29 to 47%)</span></div>
    <div class="step gate"><b>5. Track</b><span>SAM2 mask over 10 frames each side, classify every frame, fuse</span></div>
  </div>
  <p class="note">Steps 4 and 5 use future frames, so the pipeline is offline. TAPIS is also non-causal (its clip spans about 8 s).</p>
</section>

<section>
  <h2>Result against TAPIS</h2>
  <div class="tablewrap"><table>
    <tr><th>Method</th><th class="n">mIoU</th><th class="n">IoU</th><th class="n">mcIoU</th></tr>
    <tr class="ours"><td>This pipeline, seed 44 (seed chosen in advance on a held-out fold)</td><td class="n">87.49</td><td class="n">86.41</td><td class="n">79.78</td></tr>
    <tr class="ours"><td>This pipeline, mean of 3 classifier seeds</td><td class="n">87.37 ± 0.33</td><td class="n">86.22 ± 0.46</td><td class="n">78.33 ± 1.28</td></tr>
    <tr class="sub"><td>95% interval, bootstrap over the 5 test cases</td><td class="n">86.85 to 88.21</td><td class="n">85.56 to 87.09</td><td class="n">76.33 to 79.81</td></tr>
    <tr class="sub"><td>Share of frame resamples above TAPIS</td><td class="n">97%</td><td class="n">100%</td><td class="n">81%</td></tr>
    <tr><td>TAPIS (Mask2Former Swin-L + video transformer), published</td><td class="n">86.61</td><td class="n">83.38</td><td class="n">77.42</td></tr>
    <tr><td>TAPIS-VST, published</td><td class="n">86.36</td><td class="n">83.51</td><td class="n">77.54</td></tr>
    <tr><td>SlowFast, published</td><td class="n">77.16</td><td class="n">72.26</td><td class="n">58.75</td></tr>
  </table></div>
  <div class="chartbox" style="margin-top:12px">
    __CHART1__
    <div class="legend">
      <span><i style="background:var(--accent);border-radius:50%"></i>ours, 3-seed mean, whisker = 95% case bootstrap</span>
      <span><i style="background:var(--tapis)"></i>TAPIS (square)</span>
      <span><i style="border:2px solid var(--tapis);border-radius:2px"></i>TAPIS-VST (triangle); both values under the markers</span>
    </div>
  </div>
  <p class="note">Reading: IoU clearly above TAPIS, mIoU modestly above, mcIoU not distinguishable (TAPIS lies inside our interval). ISINet was shelved; it is only published on a different split.</p>
</section>

<section>
  <h2>What each step added</h2>
  <p class="note" style="margin-top:0">3-seed means, official test, 833 instruments tracked.</p>
  <div class="tablewrap"><table>
    <tr><th>Step</th><th class="n">mIoU</th><th class="n">IoU</th><th class="n">mcIoU</th></tr>
    <tr><td>SAM2 fine-tuned, checkpoint chosen on the test cases (first run)</td><td class="n">84.86</td><td class="n">83.68</td><td class="n">74.44</td></tr>
    <tr><td>SAM2 fine-tuned, checkpoint chosen on a held-out fold</td><td class="n">86.34</td><td class="n">85.31</td><td class="n">77.59</td></tr>
    <tr><td>+ SAM3 added to the segmenter</td><td class="n">86.78</td><td class="n">85.72</td><td class="n">77.90</td></tr>
    <tr><td>+ classifier trained on tracker-style crops</td><td class="n">87.11</td><td class="n">85.97</td><td class="n">78.19</td></tr>
    <tr class="ours"><td>+ segmenters trained on all 8 training cases (final)</td><td class="n">87.37</td><td class="n">86.22</td><td class="n">78.33</td></tr>
    <tr class="sub"><td>Ceiling: perfect classes on these masks</td><td class="n">91.17</td><td class="n">91.17</td><td class="n">88.95</td></tr>
  </table></div>
  <p class="note">Steps of 0.3 or less are inside the seed noise. Mean mask IoU against ground truth: 0.9077 (final test masks).</p>

  <div style="height:22px"></div>
  <p class="note" style="margin-top:0">Effect of tracking, final configuration:</p>
  <div class="tablewrap"><table>
    <tr><th>Tracking</th><th class="n">mIoU</th><th class="n">IoU</th><th class="n">mcIoU</th><th class="n">accuracy</th><th class="n">macro-F1</th></tr>
    <tr><td>None (single pass)</td><td class="n">83.47</td><td class="n">81.07</td><td class="n">69.94</td><td class="n">0.909</td><td class="n">0.863</td></tr>
    <tr><td>539 most uncertain (19%)</td><td class="n">86.52</td><td class="n">85.11</td><td class="n">76.69</td><td class="n">0.948</td><td class="n">0.917</td></tr>
    <tr class="ours"><td>833 most uncertain (29%), registered headline</td><td class="n">87.37</td><td class="n">86.22</td><td class="n">78.33</td><td class="n">0.957</td><td class="n">0.929</td></tr>
    <tr><td>Held-out threshold (tracks 42 to 47%)</td><td class="n">87.50</td><td class="n">86.43</td><td class="n">78.90</td><td class="n">0.958</td><td class="n">0.932</td></tr>
  </table></div>
  <div class="chartbox" style="margin-top:12px">__CHART3__</div>
  <p class="note">The 833 comes from an earlier gate that was picked on the test set. The threshold chosen on a held-out fold scores the same or higher and tracks more.</p>
</section>

<section>
  <h2>Per class (IoU, final)</h2>
  <div class="chartbox">__CHART2__</div>
  <p class="note">Laparoscopic Grasper is the weak class: with closed jaws it looks like a suction tube in a single frame. Tracking cannot separate them.</p>
</section>

<section>
  <h2>Classifier history</h2>
  <p class="note" style="margin-top:0">Per-instrument classification on ground-truth masks, official test. Earlier, vote-based pipeline.</p>
  <div class="tablewrap"><table>
    <tr><th>Configuration</th><th class="n">accuracy</th><th class="n">macro-F1</th></tr>
    <tr><td>Whole-frame label (exact match)</td><td class="n">0.359</td><td class="n">0.708</td></tr>
    <tr><td>Single MobileNetV3-small</td><td class="n">0.867</td><td class="n">0.834</td></tr>
    <tr><td>Single ResNet-50, 320 px</td><td class="n">0.915</td><td class="n">0.865</td></tr>
    <tr><td>4-model ensemble</td><td class="n">0.927</td><td class="n">0.890</td></tr>
    <tr><td>+ uncertainty-gated SAM2 tracking</td><td class="n">0.962</td><td class="n">0.935</td></tr>
  </table></div>
  <p class="note">The shipped uncertainty method is now a single-pass evidential ensemble instead of 80 stochastic votes: 1/80 of the cost, 0.3 to 0.5 points lower accuracy over 3 seeds.</p>
</section>

<section>
  <h2>Uncertainty: softmax, MC dropout, evidential</h2>
  <p class="note" style="margin-top:0">Three ways to score how likely a prediction is wrong, compared on a held-out fold (3 seeds each, mean ± SD) and on the official test.</p>
  <div class="tablewrap"><table>
    <tr><th>Signal</th><th class="n">AUROC fold1</th><th class="n">AUROC test</th><th class="n">errors caught at 20% review</th><th class="n">unanimous errors flagged</th><th class="n">passes per instrument</th></tr>
    <tr><td>Softmax confidence</td><td class="n">0.927 ± 0.009</td><td class="n">0.916</td><td class="n">86.6%</td><td class="n">0%</td><td class="n">1</td></tr>
    <tr><td>MC-dropout vote disagreement</td><td class="n">0.912 ± 0.010</td><td class="n">0.896</td><td class="n">83.5%</td><td class="n">0%</td><td class="n">80</td></tr>
    <tr class="ours"><td>Evidential (Dirichlet, epistemic variance)</td><td class="n">0.927 ± 0.004</td><td class="n">0.922</td><td class="n">89.3%</td><td class="n">74%</td><td class="n">1</td></tr>
  </table></div>
  <p class="note">AUROC: 0.5 is chance, 1.0 ranks every wrong prediction above every right one. "Unanimous" errors are wrong predictions where every member agrees on the same wrong class, so a vote or a softmax shows no disagreement and cannot flag them; evidential also looks at how much total evidence the model has. Test AUROC for evidential is the mean of 3 seeds (0.914 to 0.928); the softmax and MC rows come from the vote-pipeline ensemble.</p>

  <div style="height:18px"></div>
  <div class="tablewrap"><table>
    <tr><th>Why evidential</th><th>Evidence</th><th></th></tr>
    <tr><td>As good at finding errors</td><td>AUROC matches softmax and beats MC dropout by 1.5 points on fold1, with lower seed-to-seed spread</td><td><span class="chip good">for</span></td></tr>
    <tr><td>80 times cheaper than MC dropout</td><td>1 forward pass instead of 4 members x 20 stochastic passes</td><td><span class="chip good">for</span></td></tr>
    <tr><td>Catches errors the others cannot</td><td>flags 74% of unanimous errors at 20% review; softmax and vote flag none by construction</td><td><span class="chip good">for</span></td></tr>
    <tr><td>Defensible as an uncertainty measure</td><td>variance-based Dirichlet evidence (Duan et al., WACV 2024); softmax confidence is a contested uncertainty score, so it is only a baseline here</td><td><span class="chip good">for</span></td></tr>
    <tr><td>Accuracy</td><td>0.3 to 0.5 points below the vote pipeline over 3 seeds</td><td><span class="chip warn">cost</span></td></tr>
    <tr><td>Against softmax</td><td>AUROC is equal (0.9271 vs 0.9274); the case for evidential over softmax is defensibility and the unanimous errors, not accuracy</td><td><span class="chip warn">cost</span></td></tr>
    <tr><td>Stability under domain shift</td><td>EndoVis zero-shot: evidential flags about 85% of instruments, vote about 80%. No benefit; no signal transfers its threshold without retraining</td><td><span class="chip bad">no gain</span></td></tr>
  </table></div>
</section>

<section>
  <h2>How the gate was chosen</h2>
  <p class="note" style="margin-top:0">The gate decides which instruments get the expensive SAM2 tracking. Rule fixed before running: train on one fold, sweep the threshold on the other fold with tracking, take the highest accuracy, then the highest threshold within 0.001 of it (the cheaper gate). The official test is not touched.</p>
  <div class="chartbox">__CHART4__</div>
  <div style="height:14px"></div>
  <div class="tablewrap"><table>
    <tr><th>Gate</th><th>Chosen on</th><th class="n">test tracked</th><th class="n">accuracy</th><th class="n">macro-F1</th></tr>
    <tr><td>Vote disagreement at 9%</td><td><span class="chip bad">the test set</span></td><td class="n">833</td><td class="n">0.9623</td><td class="n">0.9351</td></tr>
    <tr><td>Vote disagreement, re-chosen at 15%</td><td><span class="chip good">fold1</span></td><td class="n">649</td><td class="n">0.9595</td><td class="n">0.9309</td></tr>
    <tr><td>Evidential, threshold 1.7e-5 (earlier members, seed 42; other seeds flag 528 and 552)</td><td><span class="chip good">fold1</span></td><td class="n">539</td><td class="n">0.9507</td><td class="n">0.9279</td></tr>
    <tr class="ours"><td>Final pipeline, same threshold (tracks 42 to 47%)</td><td><span class="chip good">fold1</span></td><td class="n">42 to 47% of 2,861</td><td class="n">0.958</td><td class="n">0.932</td></tr>
  </table></div>
  <p class="note">The accuracy curve is flat beyond about 15% tracked (0.9589 at 15%, 0.9604 at 20%, 0.9607 at 31%), so the exact threshold matters little. The test-chosen 9% vote gate was about 0.3 points optimistic against the same rule applied on fold1. With the final tracker-style-crop members the same threshold flags 42 to 47% of test instruments, because they are less confident; that is why the headline tracks a fixed 833 instead.</p>
</section>

<section>
  <h2>Cost</h2>
  <p class="note" style="margin-top:0">One Titan Xp (Pascal, no fast half precision), one frame at a time, 2.5 instruments per frame.</p>
  <div class="tablewrap"><table>
    <tr><th>Stage</th><th class="n">time</th></tr>
    <tr><td>SAM2 masks (with mirror pass)</td><td class="n">0.53 s (1.05 s)</td></tr>
    <tr><td>SAM3 masks (with mirror pass)</td><td class="n">1.16 s (2.38 s)</td></tr>
    <tr><td>Four-member classifier</td><td class="n">0.10 s per frame</td></tr>
    <tr><td>SAM2 tracking</td><td class="n">about 13 s per tracked instrument</td></tr>
    <tr class="ours"><td>TAPIS, for reference</td><td class="n">0.48 s per keyframe</td></tr>
  </table></div>
  <p class="note">The full pipeline (SAM2 + SAM3, SAM2-large tracking) is an offline batch tool, far slower than TAPIS. The real-time variant is below.</p>
</section>

<section>
  <h2>Real-time version (causal, 1 frame per second)</h2>
  <p class="note" style="margin-top:0">Past frames only, one Titan Xp, ground-truth boxes. Settings fixed in a protocol addendum before the runs (the YOLO rung was added afterwards, at your request, with its earlier locked settings). 3-seed means, official test, unclipped.</p>
  <div class="tablewrap"><table>
    <tr><th>Pipeline</th><th class="n">tracked</th><th class="n">mIoU</th><th class="n">IoU</th><th class="n">mcIoU</th><th class="n">inst. acc.</th></tr>
    <tr class="ours"><td>Tiny + flip, causal EdgeTAM (best real-time)</td><td class="n">539</td><td class="n">84.22</td><td class="n">82.30</td><td class="n">71.89</td><td class="n">0.929</td></tr>
    <tr><td>Tiny + flip, causal EdgeTAM, wider gate</td><td class="n">1,014</td><td class="n">84.54</td><td class="n">82.76</td><td class="n">72.44</td><td class="n">0.933</td></tr>
    <tr><td>Tiny + flip, causal YOLO26s</td><td class="n">539</td><td class="n">83.23</td><td class="n">81.32</td><td class="n">70.28</td><td class="n">0.917</td></tr>
    <tr><td>Tiny + flip, causal YOLO26s, its own gate</td><td class="n">888</td><td class="n">83.34</td><td class="n">81.55</td><td class="n">70.43</td><td class="n">0.918</td></tr>
    <tr><td>Large, no flip, causal EdgeTAM</td><td class="n">539</td><td class="n">83.62</td><td class="n">81.62</td><td class="n">71.18</td><td class="n">0.923</td></tr>
    <tr><td>Tiny + flip, no tracking</td><td class="n">0</td><td class="n">82.13</td><td class="n">79.55</td><td class="n">67.93</td><td class="n">0.90</td></tr>
    <tr><td>Large, no flip, no tracking</td><td class="n">0</td><td class="n">81.45</td><td class="n">78.80</td><td class="n">66.79</td><td class="n">0.90</td></tr>
    <tr class="sub"><td>Offline final (non-causal SAM2-large tracking)</td><td class="n">833</td><td class="n">87.37</td><td class="n">86.22</td><td class="n">78.33</td><td class="n">0.957</td></tr>
    <tr class="sub"><td>TAPIS, published</td><td class="n"></td><td class="n">86.61</td><td class="n">83.38</td><td class="n">77.42</td><td class="n"></td></tr>
  </table></div>
  <p class="note">Real time lands 2.4 mIoU, 1.1 IoU and 5.5 mcIoU below TAPIS, and about 3 mIoU below the offline result. Tracking adds about 2 mIoU over no tracking; causal EdgeTAM recovers less than offline SAM2-large (+2.1 against +3.05 mIoU).</p>

  <div style="height:18px"></div>
  <p class="note" style="margin-top:0">Segmenter choice (SAM2.1 variants, fine-tuned the same way, mean mask IoU on held-out fold1, per-frame time on the Titan Xp, 2.5 instruments per frame, fp32):</p>
  <div class="tablewrap"><table>
    <tr><th>Variant</th><th class="n">IoU, no flip</th><th class="n">IoU, flip</th><th class="n">time, no flip</th><th class="n">time, flip</th></tr>
    <tr><td>Large</td><td class="n">0.9059</td><td class="n">0.9086</td><td class="n">402 ms</td><td class="n">807 ms</td></tr>
    <tr><td>Small</td><td class="n">0.9023</td><td class="n">0.9069</td><td class="n">121 ms</td><td class="n">234 ms</td></tr>
    <tr class="ours"><td>Tiny (chosen)</td><td class="n">0.9025</td><td class="n">0.9061</td><td class="n">108 ms</td><td class="n">207 ms</td></tr>
  </table></div>
  <p class="note">Rule fixed in advance: the fastest variant within 0.005 of large. Test mask IoU of the all-case models: tiny with flip 0.8984, large without flip 0.8965.</p>

  <div style="height:18px"></div>
  <div class="tablewrap"><table>
    <tr><th>Stage</th><th class="n">time</th></tr>
    <tr><td>Tiny segmenter with flip</td><td class="n">0.21 s per frame</td></tr>
    <tr><td>Four-member classifier</td><td class="n">0.10 s per frame</td></tr>
    <tr class="ours"><td>Provisional label (no tracking)</td><td class="n">about 0.31 s per frame</td></tr>
    <tr><td>YOLO26s tracker</td><td class="n">about 10 ms per frame, corrected label immediate</td></tr>
    <tr><td>EdgeTAM tracker</td><td class="n">about 2.9 s compute per refinement, corrected label about 3 s later (earlier replay)</td></tr>
  </table></div>
  <p class="note">Latency is stitched from stage timings and an earlier streaming replay, not from one end-to-end run of this pipeline. One GPU sustained about a 14% tracked share in that replay, so the 539 row (19%) needs two GPUs for EdgeTAM. The scores are for the corrected labels; the provisional label scores like the no-tracking row.</p>
  <p class="note">Also tried: fusing evidence across frames by matching boxes, with no tracker. It was 0.1 to 0.2 points worse than doing nothing on held-out fold1 (box matches at 1 frame per second are wrong 13 to 19% of the time), so it was not run on test.</p>
</section>

<section>
  <h2>Tried and not adopted</h2>
  <div class="tablewrap"><table>
    <tr><th>Idea</th><th>Result</th><th>Verdict</th></tr>
    <tr><td>Mask refinement network</td><td>fold1 mask IoU 0.9120 vs SAM 0.9086; box clipping alone 0.9123</td><td><span class="chip bad">rejected</span></td></tr>
    <tr><td>SAM3 alone, zero-shot</td><td>0.8624 vs fine-tuned SAM2 0.9109 (fold1 sample)</td><td><span class="chip bad">rejected</span></td></tr>
    <tr><td>SAM3 fine-tuned, ensembled with SAM2</td><td>fold1 0.9131 vs 0.9086 SAM2 alone; missed the pre-set bar of 0.9136 by 0.0005</td><td><span class="chip warn">adopted with disclosure</span></td></tr>
    <tr><td>Mask post-processing (holes, largest component, polygon, erosion)</td><td>all hurt</td><td><span class="chip bad">dropped</span></td></tr>
    <tr><td>Clip mask to its box</td><td>+0.24 mIoU, only because ground-truth boxes are tight</td><td><span class="chip neutral">reported separately</span></td></tr>
    <tr><td>Tracker-style crops in classifier training</td><td>fold1 +0.63 pts, confirmed on fold2</td><td><span class="chip good">adopted</span></td></tr>
    <tr><td>Perturbed-mask training</td><td>0.9267 vs 0.9274 baseline</td><td><span class="chip bad">no effect</span></td></tr>
    <tr><td>Tight crops with masking and lighting augmentation</td><td>macro-F1 0.78 to 0.79 vs 0.82 to 0.83</td><td><span class="chip bad">worse</span></td></tr>
    <tr><td>ResNet-101 member</td><td>held up alone, did not beat the ensemble</td><td><span class="chip bad">not adopted</span></td></tr>
    <tr><td>Fusing evidence across frames by box matching (real-time, no tracker)</td><td>fold1 accuracy 0.9319 vs 0.9336 unfused at best</td><td><span class="chip bad">rejected</span></td></tr>
    <tr><td>YOLO26s as the causal tracker</td><td>about 1 mIoU below EdgeTAM at 539 tracked, but about 10 ms per frame</td><td><span class="chip warn">cheaper, less accurate</span></td></tr>
    <tr><td>INT8 quantization</td><td>slower on CPU, lost 1.3 to 79 points of accuracy</td><td><span class="chip bad">not adopted</span></td></tr>
  </table></div>
</section>

<section>
  <h2>Release</h2>
  <div class="tablewrap"><table>
    <tr><th>Item</th><th>State</th></tr>
    <tr><td>Code and CLI (whole pipeline)</td><td><span class="chip good">merged</span> github.com/volcanictv/grasp-instrument-classifier</td></tr>
    <tr><td>Weights (29 files, checksum-verified fetch)</td><td><span class="chip good">public</span> huggingface.co/AryanB005/grasp-instrument-pipeline</td></tr>
    <tr><td>Tests without weights</td><td><span class="chip good">24 pass</span></td></tr>
    <tr><td>Licence</td><td><span class="chip warn">not chosen</span></td></tr>
    <tr><td>SAM3 delta (derived from a gated model)</td><td><span class="chip warn">check SAM licence</span></td></tr>
  </table></div>
</section>

<section>
  <h2>Limits and open items</h2>
  <div class="tablewrap"><table>
    <tr><th>Item</th><th>Status</th></tr>
    <tr><td>Ground-truth boxes</td><td>No detector in these numbers. Next: run on TAPIS's own boxes or a trained detector.</td></tr>
    <tr><td>Segmenter seed variance</td><td>Each segmenter trained once. Seed spread above is the classifier only.</td></tr>
    <tr><td>Member weight 0.40</td><td>Set on the test set early on; held-out folds did not confirm it over flat weights. Flat-weight run not done.</td></tr>
    <tr><td>Tracking budget</td><td>Headline uses 833 tracked (from an earlier test-chosen gate); held-out threshold gives the same or better.</td></tr>
    <tr><td>mcIoU margin</td><td>Not significant against TAPIS.</td></tr>
    <tr><td>Causal operation</td><td>Offline pipeline is non-causal. The real-time version is causal and scores 84.2 / 82.3 / 71.9, below TAPIS on all three.</td></tr>
    <tr><td>Real-time latency</td><td>From stage timings and an earlier replay; no single end-to-end replay of this pipeline yet.</td></tr>
    <tr><td>One dataset, 5 test cases</td><td>Intervals are wide on mcIoU.</td></tr>
  </table></div>
</section>

<footer>
  Sources: <code>docs/reports/gtbox_sam/final_3seed.json</code>, <code>bootstrap_cis.json</code>, <code>latency/our_stages.json</code>, <code>latency/tapis.json</code>, the protocol in <code>docs/reports/gtbox_sam_protocol.md</code> (research repository, branch gtbox-sam-pipeline, merged), and the earlier status report of 2026-09-20 for the classifier history. TAPIS numbers are from the GraSP paper.
</footer>

</main>
"""

HTML = HTML.replace("__CHART1__", CHART1).replace("__CHART2__", CHART2).replace("__CHART3__", CHART3).replace("__CHART4__", CHART4)
assert "xx" not in HTML
assert "—" not in HTML and "&mdash;" not in HTML
OUT.write_text(HTML, encoding="utf-8")
print("wrote", OUT, len(HTML))
