"""Pipeline-level cost in GFLOPs (multiply-adds) from the per-stage counts of scripts/count_gflops.py, next to the measured A100 times.

Per instrument:  single pass  = (mirror-averaged SAM2-large + SAM3 on a frame, shared by the instruments of the frame, plus one box decoding each) / instruments per frame + the four classifier members
                 refinement  = SAM2-large tracking over 21 frames + 21 classifications by the four members
                 average     = single pass + share refined * refinement
Usage:  python scripts/gflops_table.py --out docs/reports/paper_gflops.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "docs" / "reports"
INSTR_PER_FRAME = 2861 / 1125


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    g = json.loads((R / "gflops_main.json").read_text())
    s3 = json.loads((R / "gflops_sam3.json").read_text())["sam3"]
    lat = json.loads((R / "latency_a100" / "paper_table.json").read_text())["Non-causal, SAM2 + SAM3 masks, SAM2-large tracking"]
    sam2l, sam2t = g["sam2_large"], g["sam2_tiny"]
    n = INSTR_PER_FRAME
    seg_frame = 2 * (sam2l["one_pass_one_box"] - sam2l["per_extra_box"]) + 2 * (s3["one_pass_one_box"] - s3["per_extra_box"])  # image encoders, mirror pass included
    seg_boxes = n * 2 * (sam2l["per_extra_box"] + s3["per_extra_box"])
    seg_instrument = (seg_frame + seg_boxes) / n
    cls = g["classifier_four_members"]
    single = seg_instrument + cls
    refine = g["refined_instrument"]
    tiny_single = (2 * (sam2t["one_pass_one_box"] - sam2t["per_extra_box"]) + n * 2 * sam2t["per_extra_box"]) / n + cls
    out = {"convention": g["convention"], "instruments_per_frame": n,
           "stages_gflops": {"SAM2.1-large, one pass": sam2l["one_pass_one_box"], "SAM2.1-tiny, one pass": sam2t["one_pass_one_box"], "SAM3, one pass": s3["one_pass_one_box"],
                             "classifier, four members, one crop": cls, "SAM2-large tracking, 21 frames": g["sam2_large_tracking_21_frames"], "refinement of one instrument": refine},
           "params_m": {"SAM2.1-large": sam2l["params_m"], "SAM2.1-tiny": sam2t["params_m"], "SAM3": s3["params_m"], "classifier members": sum(v["params_m"] for v in g["classifier_members"].values())},
           "single_pass_per_instrument": single, "single_pass_per_frame": seg_frame + seg_boxes + n * cls, "tiny_segmenter_single_pass_per_instrument": tiny_single, "average": {}, "a100_seconds": {
               "single_pass_per_frame": lat["frame_stage_per_instrument_ms"] / 1000, "refinement_per_instrument": lat["cost_per_tracked_ms"] / 1000}}
    for name, share in (("29% refined", 0.29), ("held-out threshold, 45% refined", 0.45), ("every instrument refined", 1.0)):
        avg = single + share * refine
        out["average"][name] = {"share": share, "gflops": avg, "saving_vs_all": 1 - avg / (single + refine)}
        print(f"{name:34s} {avg / 1000:6.2f} TFLOPs per instrument  (saving against refining all {100 * (1 - avg / (single + refine)):.0f}%)")
    print(f"single pass per instrument {single / 1000:.2f} TFLOPs, refinement per instrument {refine / 1000:.2f} TFLOPs, tiny-segmenter single pass {tiny_single / 1000:.2f} TFLOPs")
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
