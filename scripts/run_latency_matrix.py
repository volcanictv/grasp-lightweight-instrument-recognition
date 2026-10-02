"""Cross-platform version of scripts/run_latency_matrix.sh: frame-level latency of the three trackers on a busy
segment, every refinement finished before the next frame (--sync-refine), so a frame's latency is the full cost of
ingesting it with all its instruments as one unit. Settings per tracker: natural (official gate), share13 (gate
set so about 13% of instruments fire; pass the threshold measured on the reference machine so machines are
compared on identical work) and allgated (gate 0, the worst case).

Usage (laptop, from the repo root):
    python scripts/run_latency_matrix.py --segment C:/Users/aryan/bench/segment --yolo-weights C:/Users/aryan/bench/yolo26s_official_last.pt \\
        --edgetam-repo C:/Users/aryan/bench/EdgeTAM --sam2-repo C:/Users/aryan/bench/sam2 --tau13 2.7642995387644804e-05 \\
        --out-dir experiments/rt_latency_laptop
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", type=Path, required=True)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--edgetam-repo", type=Path, required=True)
    ap.add_argument("--sam2-repo", type=Path, required=True)
    ap.add_argument("--tau13", type=float, required=True)
    ap.add_argument("--base-config", type=Path, default=REPO_ROOT / "configs" / "realtime_grasp_edgetam.yaml")
    ap.add_argument("--trackers", default="match,edgetam,sam2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.base_config.read_text())
    plans = {
        "match": dict(sam_config="configs/edgetam.yaml", ckpt=None, repo=None, all_frames=90),
        "edgetam": dict(sam_config="configs/edgetam.yaml", ckpt=args.edgetam_repo / "checkpoints" / "edgetam.pt", repo=args.edgetam_repo, all_frames=40),
        "sam2": dict(sam_config="configs/sam2.1/sam2.1_hiera_l.yaml", ckpt=args.sam2_repo / "checkpoints" / "sam2.1_hiera_large.pt", repo=args.sam2_repo, all_frames=20),
    }
    settings = {"natural": (0.000285, 90), "share13": (args.tau13, 90), "allgated": (0.0, None)}
    for tracker in args.trackers.split(","):
        plan = plans[tracker]
        for setting, (tau, frames) in settings.items():
            cfg = yaml.safe_load(yaml.safe_dump(base))
            cfg["tracker"].update(name=tracker, sam_config=plan["sam_config"])
            cfg["gate"]["threshold"] = tau
            cfg_path = args.out_dir / f"cfg_{tracker}_{setting}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))
            out = args.out_dir / f"{tracker}_{setting}.json"
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "realtime_stream_test.py"), "--config", str(cfg_path),
                   "--frames-dir", str(args.segment), "--yolo-weights", str(args.yolo_weights), "--device", args.device,
                   "--sync-refine", "--max-frames", str(frames or plan["all_frames"]), "--out", str(out)]
            if plan["ckpt"]:
                cmd += ["--sam-checkpoint", str(plan["ckpt"])]
            env = dict(os.environ)
            if plan["repo"]:
                env["PYTHONPATH"] = str(plan["repo"]) + os.pathsep + env.get("PYTHONPATH", "")
            print(f"run {tracker} {setting}", flush=True)
            with open(args.out_dir / f"{tracker}_{setting}.log", "w") as log:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode
            print(f"  exit {code}", flush=True)


if __name__ == "__main__":
    main()
