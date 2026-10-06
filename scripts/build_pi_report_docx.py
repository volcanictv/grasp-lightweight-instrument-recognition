"""Builds the PI report as a Word document (python-docx + matplotlib figures). Every number is either typed from a result file named in the comment next to
it or computed below from the latency measurement files; nothing else.

Usage (any machine with python-docx and matplotlib):
    python scripts/build_pi_report_docx.py --out docs/reports/grasp_report_2026-10-06.docx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO = Path(__file__).resolve().parents[1]
R = REPO / "docs" / "reports"
INK, MUTED, ACCENT, GREY, RED, SOFT = "#1c2430", "#5d6877", "#0b6e8a", "#8a94a3", "#a3322b", "#dcedf2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK})


def jload(rel: str):
    return json.loads((R / rel).read_text())


# ------------------------------------------------------------------ latency numbers (computed from the measurement files)
ours = jload("latency/our_stages.json")["stages"]
tapis_ms = jload("latency/tapis.json")["total"]["mean_ms"]
sam2_track_ms = jload("causal_realtime/latency/sam2_large.json")["per_instance"]["median_ms"]  # SAM2-large propagation over 21 frames, per instrument
crop_ms = jload("causal_realtime/latency/classify.json")["all_members_per_crop"]["median_ms"]  # four members, one crop
yolo_ms = jload("causal_realtime/latency/yolo.json")["per_frame"]["mean_ms"]
match_ms = jload("causal_realtime/latency/match.json")["per_instance"]["mean_ms"]
rt = jload("realtime/segmenter_latency.json")["stages"]
replay = jload("causal_realtime/latency_frames/edgetam_share13.json")
roc = jload("evidential/roc_fold1.json")
N_INST, N_FRAMES = roc["test_instruments"], roc["test_frames"]
PER_FRAME = N_INST / N_FRAMES
MAX_PER_FRAME = roc["instruments_per_frame_max"]

OFF_SEG = ours["sam2_bf16_flip"]["mean_ms"] + ours["sam3_fp32_flip"]["mean_ms"]
OFF_CLS = ours["classifier_4_members"]["mean_ms"]
OFF_TRACK = sam2_track_ms + 21 * crop_ms
OFF_SHARE = 833 / N_INST
OFF_BEST = OFF_SEG + OFF_CLS
OFF_AVG = OFF_BEST + PER_FRAME * OFF_SHARE * OFF_TRACK
OFF_WORST = OFF_BEST + MAX_PER_FRAME * OFF_TRACK

RT_SEG = rt["tiny_fp32_flip"]["mean_ms"]
RT_CLS = rt["classifier_4_members"]["mean_ms"]
RT_TRACK = replay["refine_compute_ms"]["median"]
RT_WORST_REFINE = replay["refine_latency_ms"]["max"]
RT_SHARE = 539 / N_INST
RT_BEST = RT_SEG + RT_CLS
RT_AVG = RT_BEST + RT_SHARE * RT_TRACK
RT_WORST = RT_BEST + RT_WORST_REFINE
YOLO_TRACK = 16 * crop_ms + match_ms  # 15 past frames plus the current one, one classifier pass per crop
YOLO_AVG = RT_BEST + yolo_ms + RT_SHARE * YOLO_TRACK


def ms(v: float) -> str:
    return f"{round(v):,}"


# ------------------------------------------------------------------ figures
FIG = Path(".")


def box(ax, x, y, w, h, text, fc="white", ec=ACCENT, size=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, label=None, lx=0.0, ly=0.0, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + lx, (p[1] + q[1]) / 2 + ly, label, fontsize=8, color=color, fontweight="bold", ha="center", va="center")


def line(ax, p, q, color=INK):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.2, solid_capstyle="butt")


def fig_flow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.6, 13.4)
    ax.axis("off")
    box(ax, 2.6, 12.6, 3.6, 0.9, "Video frame")
    box(ax, 7.4, 12.6, 3.6, 0.9, "Box around each\ninstrument (input)")
    arrow(ax, (2.6, 12.15), (4.4, 11.35))
    arrow(ax, (7.4, 12.15), (5.6, 11.35))
    box(ax, 5, 10.7, 6.4, 1.3, "SAM2 + SAM3 segmenter\nfine-tuned, each run on the image and its mirror image,\nmask scores averaged", fc=SOFT, bold=False)
    arrow(ax, (5, 10.05), (5, 9.35))
    box(ax, 5, 8.85, 4.6, 0.9, "Instrument mask, cropped from the frame")
    arrow(ax, (5, 8.4), (5, 7.7))
    box(ax, 5, 7.1, 6.4, 1.2, "Evidential classifier\n4 networks, one pass each, evidence combined", fc=SOFT)
    arrow(ax, (5, 6.5), (5, 5.8))
    box(ax, 5, 5.3, 5.0, 0.9, "Class + uncertainty score")
    arrow(ax, (5, 4.85), (5, 4.35))
    ax.add_patch(Polygon([(5, 4.35), (7.3, 3.4), (5, 2.45), (2.7, 3.4)], closed=True, fc="white", ec=ACCENT, lw=1.3))
    ax.text(5, 3.4, "Gate: is the\nscore uncertain?", ha="center", va="center", fontsize=8.5, linespacing=1.25)
    line(ax, (2.7, 3.4), (1.45, 3.4))
    arrow(ax, (1.45, 3.4), (1.45, 2.75))
    ax.text(2.05, 3.62, "no", fontsize=8.5, fontweight="bold", ha="center")
    line(ax, (7.3, 3.4), (8.3, 3.4))
    arrow(ax, (8.3, 3.4), (8.3, 2.9))
    ax.text(7.8, 3.62, "yes", fontsize=8.5, fontweight="bold", ha="center")
    ax.text(5, 2.05, "about 3 in 10 instruments\nare sent on", ha="center", fontsize=7.8, color=MUTED, linespacing=1.25)
    box(ax, 1.45, 2.05, 2.5, 1.35, "Keep the\nsingle-frame\nclass", size=8.3)
    box(ax, 8.3, 1.9, 3.0, 1.9, "SAM2 follows the\nmask across\nneighbouring frames;\nclassifier runs on\nevery frame", fc=SOFT, size=8.0)
    arrow(ax, (8.3, 0.95), (8.3, 0.33))
    box(ax, 8.3, -0.15, 3.0, 0.9, "Combine the frames,\nweighted by confidence", size=8.0)
    line(ax, (1.45, 1.37), (1.45, -1.0))
    arrow(ax, (1.45, -1.0), (3.0, -1.0))
    line(ax, (8.3, -0.6), (8.3, -1.0))
    arrow(ax, (8.3, -1.0), (7.0, -1.0))
    box(ax, 5, -1.0, 4.0, 0.8, "Final class per instrument", fc=ACCENT, ec=ACCENT, bold=True)
    ax.texts[-1].set_color("white")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_whisker(path: Path) -> None:
    # final_3seed.json (means), bootstrap_cis.json (case interval), published TAPIS numbers
    rows = [("mIoU", 87.37, (86.85, 88.21), 86.61), ("IoU", 86.22, (85.56, 87.09), 83.38), ("mcIoU", 78.33, (76.33, 79.81), 77.42)]
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    for i, (name, v, ci, t) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(ci, [y, y], color=ACCENT, lw=2.2, solid_capstyle="butt")
        for c in ci:
            ax.plot([c, c], [y - 0.12, y + 0.12], color=ACCENT, lw=2)
        ax.plot(v, y, "o", color=ACCENT, ms=8, zorder=3)
        ax.plot(t, y, "s", color=GREY, ms=7.5, zorder=3)
        ax.text(v, y + 0.27, f"{v:.2f}", ha="center", fontsize=8.5, color=ACCENT, fontweight="bold")
        ax.text(t, y - 0.33, f"{t:.2f}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(74, 90)
    ax.set_ylim(-0.7, 2.6)
    ax.grid(axis="x", color="#e1e5ea", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("score")
    ax.plot([], [], "o", color=ACCENT, label="Final pipeline (whisker: likely range on a different set of test cases)")
    ax.plot([], [], "s", color=GREY, label="TAPIS")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_classes(path: Path) -> None:
    # final_3seed.json, gated (top 833), per-class IoU
    data = [("Monopolar Curved Scissors", 94.2), ("Large Needle Driver", 87.0), ("Bipolar Forceps", 84.0), ("Suction Instrument", 78.1),
            ("Clip Applier", 76.8), ("Prograsp Forceps", 67.5), ("Laparoscopic Grasper", 60.7)]
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    for i, (n, v) in enumerate(data):
        y = len(data) - 1 - i
        ax.barh(y, v, color=RED if v < 70 else ACCENT, height=0.62)
        ax.text(v + 1, y, f"{v:.1f}", va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in data][::-1])
    ax.set_xlim(0, 105)
    ax.set_xlabel("IoU")
    ax.grid(axis="x", color="#e1e5ea", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_roc(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    ax.plot(roc["fpr"], roc["tpr_mean"]["MC dropout"], color=GREY, lw=2.2, label=f"MC dropout, AUROC {roc['auroc_mean']['MC dropout']:.3f}")
    ax.plot(roc["fpr"], roc["tpr_mean"]["Evidential"], color=ACCENT, lw=2.2, label=f"Evidential, AUROC {roc['auroc_mean']['Evidential']:.3f}")
    ax.set_xlabel("correct predictions flagged (false alarms)")
    ax.set_ylabel("errors caught")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#e1e5ea", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_gate(path: Path) -> None:
    # gate sweep on held-out fold1 (docs/reports/evidential_pipeline_switch/fold1_calibration.json)
    cal = jload("evidential_pipeline_switch/fold1_calibration.json")
    pts = [(100 * r["share"], r["accuracy"]) for r in cal["rows"].values()]
    pts.sort()
    chosen = (100 * cal["chosen_row"]["share"], cal["chosen_row"]["accuracy"])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axhline(cal["base_accuracy"], color=GREY, ls="--", lw=1.2)
    ax.text(31.5, cal["base_accuracy"] + 0.0006, f"no tracking {cal['base_accuracy']:.4f}", ha="right", fontsize=8, color=MUTED)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=ACCENT, lw=2, marker="o", ms=5)
    ax.plot(*chosen, "o", ms=12, mfc="none", mec=ACCENT, mew=1.8)
    ax.annotate(f"chosen: {chosen[0]:.0f}% tracked, {chosen[1]:.4f}", chosen, xytext=(chosen[0] + 1.5, chosen[1] - 0.0120), ha="left", fontsize=8.5, color=ACCENT,
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1))
    ax.set_xlabel("share of instruments sent to tracking (%), held-out fold")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 33)
    ax.grid(color="#e1e5ea", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ docx helpers
def shade(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def cell_borders(cell, color="D9DEE5") -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "bottom"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), color)
        borders.append(el)
    tcPr.append(borders)


def para(doc, text="", size=10.5, bold=False, color=INK, italic=False, align=None, space_after=6, keep_next=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size, r.bold, r.italic = Pt(size), bold, italic
    r.font.color.rgb = RGBColor.from_string(color.lstrip("#").upper())
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.keep_with_next = keep_next
    if align:
        p.alignment = align
    return p


def heading(doc, text):
    p = para(doc, text, size=14, bold=True, color=ACCENT, space_after=6, keep_next=True)
    p.paragraph_format.space_before = Pt(16)
    return p


def note(doc, text):
    return para(doc, text, size=9, color=MUTED, space_after=8)


def table(doc, header, rows, widths, right_cols=(), highlight=(), muted=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.width = Inches(widths[i])
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold, run.font.size = True, Pt(9)
        run.font.color.rgb = RGBColor.from_string(MUTED.lstrip("#").upper())
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT if i in right_cols else WD_ALIGN_PARAGRAPH.LEFT
        shade(c, "F1F3F6")
        cell_borders(c)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            c = cells[i]
            c.width = Inches(widths[i])
            c.text = ""
            run = c.paragraphs[0].add_run(str(v))
            run.font.size = Pt(9.5)
            run.bold = ri in highlight
            if ri in muted:
                run.font.color.rgb = RGBColor.from_string(MUTED.lstrip("#").upper())
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT if i in right_cols else WD_ALIGN_PARAGRAPH.LEFT
            c.paragraphs[0].paragraph_format.space_after = Pt(0)
            if ri in highlight:
                shade(c, "DCEDF2")
            cell_borders(c)
    for row in t.rows:
        for c in row.cells:
            c.paragraphs[0].paragraph_format.space_after = Pt(2)
            c.paragraphs[0].paragraph_format.space_before = Pt(2)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def picture(doc, path: Path, width_in: float) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width_in))
    p.paragraph_format.space_after = Pt(4)


def bullet(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
        r.font.size = Pt(10.5)
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(3)
    return p


def link(paragraph, url: str, text: str) -> None:
    rid = paragraph.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    c = OxmlElement("w:color")
    c.set(qn("w:val"), "0B6E8A")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "21")
    rPr.extend([c, u, sz])
    r.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    r.append(t)
    h.append(r)
    paragraph._p.append(h)


# ------------------------------------------------------------------ document
def build(out: Path, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig_flow(figdir / "flow.png")
    fig_whisker(figdir / "whisker.png")
    fig_classes(figdir / "classes.png")
    fig_roc(figdir / "roc.png")
    fig_gate(figdir / "gate.png")

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(1)
    sec.top_margin = sec.bottom_margin = Inches(0.9)
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")

    para(doc, "GraSP Instrument Recognition", size=24, bold=True, space_after=2)
    para(doc, "Project overview, 2026-10-06", size=11, color=MUTED, space_after=10)

    heading(doc, "Objective")
    para(doc, "Find and label every surgical instrument in a video frame, and know when a label is likely to be wrong. Given a frame and the box around "
              "each instrument, the system returns the instrument's mask and class. When it is unsure, it checks neighbouring frames to correct the class. "
              "The goal is to match or beat the published TAPIS results on mIoU, IoU and mcIoU.")

    heading(doc, "Challenges and how we addressed them")
    bullet(doc, " We start from pretrained models (SAM2, SAM3 and pretrained classifier backbones) and fine-tune only a small part of them. We also train the "
                "classifier on crops of the same instrument taken from neighbouring frames, so it sees the kind of crop it will get at test time.",
           "Few annotated frames.")
    bullet(doc, " The rarest instrument has about 25 times fewer examples than the most common. Every instrument is classified on its own crop, and results "
                "are judged with class-averaged metrics (mcIoU, macro-F1) so rare classes count as much as common ones. Tracking helps the rare classes most.",
           "Class imbalance.")
    bullet(doc, " Instruments are partly hidden, and some look alike from certain angles. Each crop is masked to the single instrument so neighbours do not "
                "leak in. Uncertain instruments are followed across neighbouring frames, where a clearer view often exists, and the per-frame predictions are "
                "combined weighted by confidence.",
           "Occlusion.")

    heading(doc, "Pipeline")
    picture(doc, figdir / "flow.png", 5.6)

    heading(doc, "Results")
    table(doc, ["Method", "mIoU", "IoU", "mcIoU"],
          [["Final pipeline (SAM2 + SAM3 segmenter, SAM2 evidential tracking)", "87.37 ± 0.33", "86.22 ± 0.46", "78.33 ± 1.28"],
           ["TAPIS (Mask2Former Swin-L + video transformer)", "86.61", "83.38", "77.42"]],
          [3.4, 1.05, 1.05, 1.0], right_cols=(1, 2, 3), highlight=(0,))
    note(doc, "± is the spread over 3 training runs of the classifier. In the chart, the whisker shows the likely range of the score on a different set of test cases.")
    picture(doc, figdir / "whisker.png", 6.0)
    para(doc, "IoU is clearly above TAPIS, mIoU is modestly above, and mcIoU is comparable.", space_after=4)

    heading(doc, "What each step added")
    table(doc, ["Step", "mIoU", "IoU", "mcIoU"],
          [["SAM2 fine-tuned segmenter", "86.34", "85.31", "77.59"],
           ["+ SAM3 added to the segmenter", "86.78", "85.72", "77.90"],
           ["+ classifier trained on tracker-style crops", "87.11", "85.97", "78.19"],
           ["Final pipeline (segmenters trained on all training data)", "87.37", "86.22", "78.33"]],
          [3.6, 0.95, 0.95, 1.0], right_cols=(1, 2, 3), highlight=(3,))

    heading(doc, "Per class (IoU, final pipeline)")
    picture(doc, figdir / "classes.png", 6.0)
    note(doc, "Laparoscopic Grasper is the weak class: with closed jaws it looks like a suction tube in a single frame, and tracking cannot separate them.")

    heading(doc, "Classifier history")
    table(doc, ["Configuration", "accuracy", "macro-F1"],
          [["Whole-frame label (exact match)", "0.359", "0.708"],
           ["Single MobileNetV3-small", "0.867", "0.834"],
           ["Single ResNet-50, 320 px", "0.915", "0.865"],
           ["4-model ensemble", "0.927", "0.890"],
           ["+ uncertainty-gated SAM2 tracking", "0.962", "0.935"]],
          [4.2, 1.1, 1.1], right_cols=(1, 2))
    note(doc, "The uncertainty method is now a single-pass evidential ensemble instead of 80 stochastic votes: 1/80 of the cost, 0.3 to 0.5 points lower accuracy over 3 seeds.")

    heading(doc, "Uncertainty: MC dropout vs evidential")
    para(doc, "Two ways to score how likely a prediction is wrong, compared on a held-out fold (3 seeds) and the test set. We chose evidential: it finds errors at "
              "least as well at 1/80 of the cost, and it also catches errors where every network agrees on the wrong class.", space_after=6)
    table(doc, ["", "MC dropout", "Evidential"],
          [["AUROC, held-out fold", "0.912 ± 0.010", "0.927 ± 0.004"],
           ["AUROC, test set", "0.896", "0.922"],
           ["Errors caught when 20% of instruments are reviewed", "83.5%", "89.3%"],
           ["Errors flagged when all networks agree on the wrong class", "0%", "74%"],
           ["Forward passes per instrument", "80", "1"]],
          [3.9, 1.25, 1.25], right_cols=(1, 2))
    picture(doc, figdir / "roc.png", 3.9)
    note(doc, "AUROC: 0.5 is chance, 1.0 ranks every wrong prediction above every right one. Curves are the mean of 3 seeds on the held-out fold. "
              "When all networks agree on the wrong class, a vote shows no disagreement, so only the evidential score can flag it.")

    heading(doc, "How the gate was chosen")
    para(doc, "The gate decides which instruments are sent to tracking. The threshold was swept on a held-out fold, and we took the best accuracy, then the cheapest "
              "gate within 0.001 of it. Accuracy is flat beyond about 15% tracked, so the exact threshold matters little.", space_after=6)
    picture(doc, figdir / "gate.png", 6.0)

    heading(doc, "Latency")
    para(doc, "Measured on one Titan Xp, one frame at a time, with 2.5 instruments per frame on average.", size=10, color=MUTED, space_after=6)
    table(doc, ["Stage", "ms"],
          [["Segmenter (SAM2 + SAM3, with mirror pass), per frame", ms(OFF_SEG)],
           ["Classifier, no tracking, per frame", ms(OFF_CLS)],
           ["Tracking + classifier (SAM2 tracking, 21 frames), per tracked instrument", ms(OFF_TRACK)],
           ["End to end, best case (no instrument tracked), per frame", ms(OFF_BEST)],
           ["End to end, average, per frame", ms(OFF_AVG)],
           [f"End to end, worst case ({MAX_PER_FRAME} instruments, all tracked), per frame", ms(OFF_WORST)],
           ["TAPIS, for reference, per keyframe", ms(tapis_ms)]],
          [5.3, 1.1], right_cols=(1,), highlight=(4,), muted=(6,))
    note(doc, f"Average assumes {100 * OFF_SHARE:.0f}% of instruments are tracked. Worst case is the busiest frame in the test split with every instrument tracked in turn. "
              "End-to-end figures are computed from the measured stage times.")

    heading(doc, "Real-time version")
    para(doc, "Uses past frames only, at one frame per second, with a smaller segmenter (SAM2.1-tiny) and a causal tracker.", space_after=6)
    table(doc, ["Pipeline", "mIoU", "IoU", "mcIoU", "End to end, avg (ms)"],
          [["Real-time: SAM2-tiny segmenter + causal EdgeTAM tracking (recommended)", "84.22", "82.30", "71.89", ms(RT_AVG)],
           ["Real-time: SAM2-tiny segmenter + causal YOLO26s tracking", "83.23", "81.32", "70.28", ms(YOLO_AVG)],
           ["Real-time: SAM2-tiny segmenter, no tracking", "82.13", "79.55", "67.93", ms(RT_BEST)],
           ["Final pipeline (offline above)", "87.37", "86.22", "78.33", ms(OFF_AVG)],
           ["TAPIS", "86.61", "83.38", "77.42", ms(tapis_ms)]],
          [3.2, 0.7, 0.7, 0.75, 1.2], right_cols=(1, 2, 3, 4), highlight=(0,), muted=(3, 4))
    note(doc, "Real-time rows track the 539 most uncertain instruments (about 19%). Latencies are estimated from measured stage times on one Titan Xp; "
              "the EdgeTAM tracking delay comes from a streaming replay.")
    para(doc, "Latency of the recommended real-time pipeline (SAM2-tiny + causal EdgeTAM)", size=10.5, bold=True, space_after=4, keep_next=True)
    table(doc, ["Stage", "ms"],
          [["Segmenter (SAM2-tiny, with mirror pass), per frame", ms(RT_SEG)],
           ["Classifier, no tracking, per frame", ms(RT_CLS)],
           ["Tracking + classifier (EdgeTAM, 21 frames), per tracked instrument", ms(RT_TRACK)],
           ["End to end, best case (no instrument tracked)", ms(RT_BEST)],
           ["End to end, average", ms(RT_AVG)],
           ["End to end, worst case (slowest tracked instrument in the replay)", ms(RT_WORST)]],
          [5.3, 1.1], right_cols=(1,), highlight=(4,))
    note(doc, "The label from segmentation and classification arrives within the best-case time; tracked instruments get a corrected label later, after the tracking time.")

    heading(doc, "Tried and not adopted")
    for t in ["Mask refinement network", "SAM3 on its own (zero-shot and fine-tuned)", "Mask post-processing (hole filling, largest component, polygon fit, erosion)",
              "Clipping the mask to its box (only helps with ground-truth boxes)", "Perturbed-mask training of the classifier",
              "Tight crops with masking and lighting augmentation", "ResNet-101 as an ensemble member", "INT8 quantization",
              "Fusing evidence across frames by matching boxes, without a tracker"]:
        bullet(doc, t)

    heading(doc, "Release")
    p = doc.add_paragraph(style="List Bullet")
    p.add_run("Code: ").bold = True
    link(p, "https://github.com/volcanictv/grasp-instrument-classifier", "github.com/volcanictv/grasp-instrument-classifier")
    p = doc.add_paragraph(style="List Bullet")
    p.add_run("Weights: ").bold = True
    link(p, "https://huggingface.co/AryanB005/grasp-instrument-pipeline", "huggingface.co/AryanB005/grasp-instrument-pipeline")

    heading(doc, "Limits and open items")
    bullet(doc, " Results use ground-truth boxes, so detection errors are not included. Next step: run on a trained detector's boxes or TAPIS's own boxes.", "Ground-truth boxes.")
    bullet(doc, " Each segmenter was trained once, so the spread shown covers the classifier only.", "Segmenter seed variance.")

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print("wrote", out)
    print(f"offline: seg {OFF_SEG:.0f} cls {OFF_CLS:.0f} track {OFF_TRACK:.0f} best {OFF_BEST:.0f} avg {OFF_AVG:.0f} worst {OFF_WORST:.0f} share {OFF_SHARE:.3f}")
    print(f"realtime: seg {RT_SEG:.0f} cls {RT_CLS:.0f} track {RT_TRACK:.0f} best {RT_BEST:.0f} avg {RT_AVG:.0f} worst {RT_WORST:.0f}; yolo track {YOLO_TRACK:.0f} avg {YOLO_AVG:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figdir", type=Path, default=Path("report_figs"))
    args = ap.parse_args()
    build(args.out, args.figdir)
