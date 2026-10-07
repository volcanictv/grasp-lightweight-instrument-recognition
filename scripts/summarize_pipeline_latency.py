"""Pipeline-level latency table from the JSON files of scripts/benchmark_pipeline_e2e.py (CPU only, any machine).

Every instrument's track was timed in the run, so the gate is applied afterwards, at the paper's operating points: the S1 cut-offs that select the 539 (real-time pipelines) or the 833 (final
pipelines) most uncertain test instruments, and the fold-1 threshold 2.85e-4 (cut-offs in the bundle's gate_tau.json). For every pipeline and every non-warm-up keyframe:
    frame stage        decode + segmenter + crops + 4 members + gate, shared by every instrument of the keyframe
    own-track latency  per instrument, the time until its final label: frame stage, plus (gated instruments only) its own tracking block (YOLO: plus the keyframe's batched detection)
                       -- the definition of the paper's latency tables
    queue latency      per keyframe, frame stage plus every gated tracking block one after another: the time until the last label of the keyframe
Also reported: the share of instruments the gate tracked and the per-track cost over all timed tracks.

Usage:
    python scripts/summarize_pipeline_latency.py rt_none=r/p1.json@rt rt_yolo=r/p2.json@rt noncausal_final=r/p5.json@final --tau-file bundle/gate_tau.json --out r/pipeline_latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

OPS = {"rt": ["rt_539", "tau_fold1"], "final": ["final_833", "tau_fold1"]}


def stat(v: list[float]) -> dict:
    s = sorted(v)
    return {"n": len(s), "mean_ms": statistics.mean(s), "median_ms": statistics.median(s), "p95_ms": s[int(0.95 * (len(s) - 1))], "max_ms": s[-1]} if s else {"n": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=path/to/run.json@rt or @final (the operating points of the real-time and of the final pipelines)")
    ap.add_argument("--tau-file", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    taus = json.loads(args.tau_file.read_text())
    out = {}
    for spec in args.runs:
        name, rest = spec.split("=", 1)
        path, kind = rest.rsplit("@", 1)
        r = json.loads(Path(path).read_text())
        by_kf: dict[str, list[dict]] = {}
        for rec in r["instruments"].values():
            by_kf.setdefault(rec["keyframe"], []).append(rec)
        for op in OPS[kind]:
            tau = taus[op]
            per_inst, queue, frame_ms, tracks_all, tracked, total = [], [], [], [], 0, 0
            for kf in r["keyframes"]:
                if kf["warmup"]:
                    continue
                recs = by_kf.get(kf["keyframe"], [])
                gated = [x for x in recs if x["s1"] >= tau]
                detect = kf.get("t_detect_ms", 0.0) if gated else 0.0
                own = [x["track_ms"] for x in gated]
                n = kf["n_instruments"]
                per_inst += [kf["t_frame_ms"] + detect + t for t in own] + [kf["t_frame_ms"]] * (n - len(own))
                queue.append(kf["t_frame_ms"] + detect + sum(own))
                frame_ms.append(kf["t_frame_ms"])
                tracks_all += [x["track_ms"] for x in recs]
                tracked, total = tracked + len(gated), total + n
            key = f"{name}/{op}"
            out[key] = {"gpu": r["gpu"], "tracker": r["tracker"], "causal": r["causal"], "window": r["window"], "tau": tau, "keyframes": len(queue), "instruments": total,
                        "gated_share": tracked / total if total else None, "frame_stage": stat(frame_ms), "own_track_latency_per_instrument": stat(per_inst),
                        "queue_latency_per_keyframe": stat(queue), "per_track_all_timed": stat(tracks_all)}
            o = out[key]
            print(f"{key:30s} frame {o['frame_stage']['mean_ms']:7.0f} ms | per instrument mean {o['own_track_latency_per_instrument']['mean_ms']:7.0f} ms "
                  f"(p95 {o['own_track_latency_per_instrument']['p95_ms']:7.0f}) | queue/keyframe {o['queue_latency_per_keyframe']['mean_ms']:7.0f} ms | gated {100 * (o['gated_share'] or 0):4.1f}% "
                  f"| track {o['per_track_all_timed'].get('mean_ms', float('nan')):7.0f} ms")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
