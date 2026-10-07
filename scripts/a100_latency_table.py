"""Latency of the whole pipelines on the A100, at the paper's operating points, from the measured run (docs/reports/latency_a100/run_<job>/).

The run times every instrument's tracking block, but the share of instruments its 30 or 20 keyframes happen to send to tracking differs a little from the test-set share the paper reports
(539 of 2,861 instruments for the real-time pipelines, 833 for the final ones). The average per instrument is therefore recomputed at the paper's share:
    average = frame stage + share * cost per tracked instrument            (the time until an instrument's own final label)
    queue   = frame stage + instruments per keyframe * share * cost per tracked instrument   (the time until the last label of a keyframe, tracked instruments processed one after another)
with the cost per tracked instrument measured on the instruments that pass the same S1 cut-off (YOLO: plus the batched window detection, divided over the instruments that need it).

Usage:
    python scripts/a100_latency_table.py docs/reports/latency_a100/run_21823404 --out docs/reports/latency_a100/paper_table.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

TEST_INSTRUMENTS = 2861
PIPES = [  # name, run file, gate key, budget
    ("Real time, no tracking", "p1_rt_none", None, 0),
    ("Real time, YOLO26s tracking", "p2_rt_yolo", "rt_539", 539),
    ("Real time, EdgeTAM tracking", "p3_rt_edgetam", "rt_539", 539),
    ("Causal SAM2-large tracking, tiny masks", "p4_causal_sam2", "rt_539", 539),
    ("Causal, SAM2 + SAM3 masks, SAM2-large tracking", "p6_causal_final", "final_833", 833),
    ("Non-causal, SAM2 + SAM3 masks, SAM2-large tracking", "p5_noncausal_final", "final_833", 833),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tau-file", type=Path, default=Path("bundle/gate_tau.json"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    taus = json.loads(args.tau_file.read_text())
    rows = {}
    for label, f, gate, budget in PIPES:
        r = json.loads((args.run_dir / f"{f}.json").read_text())
        kf = [k for k in r["keyframes"] if not k["warmup"]]
        frame = statistics.mean(k["t_frame_ms"] for k in kf)  # per keyframe (the queue view)
        frame_w = sum(k["n_instruments"] * k["t_frame_ms"] for k in kf) / sum(k["n_instruments"] for k in kf)  # per instrument: every instrument waits for its keyframe's frame stage
        n_per_kf = statistics.mean(k["n_instruments"] for k in kf)
        row = {"frame_stage_ms": frame, "frame_stage_per_instrument_ms": frame_w, "instruments_per_keyframe": n_per_kf, "share": budget / TEST_INSTRUMENTS}
        if gate is None:
            row.update(cost_per_tracked_ms=0.0, average_ms=frame_w, queue_ms=frame)
        else:
            tau = taus[gate]
            kfs = {k["keyframe"]: k for k in kf}
            tr = [x for x in r["instruments"].values() if x["keyframe"] in kfs and x["s1"] >= tau]
            det_kf = {x["keyframe"] for x in tr}
            cost = (sum(x["track_ms"] for x in tr) + sum(kfs[k].get("t_detect_ms", 0.0) for k in det_kf)) / len(tr)
            share = budget / TEST_INSTRUMENTS
            row.update(cost_per_tracked_ms=cost, tracked_in_sample=len(tr), average_ms=frame_w + share * cost, queue_ms=frame + n_per_kf * share * cost,
                       p95_tracked_ms=sorted(frame + x["track_ms"] for x in tr)[int(0.95 * (len(tr) - 1))])
        rows[label] = row
        print(f"{label:52s} frame {frame:6.0f}  tracked cost {row['cost_per_tracked_ms']:6.0f}  share {100 * row['share']:4.1f}%  average {row['average_ms']:6.0f}  queue {row['queue_ms']:6.0f}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
