"""Resource usage reporting for scripts/campaign.py: what stage the run is in, how much compute each task and the whole job use, and the peaks and averages.

Every 15 s the scheduler hands this module the running tasks and the latest GPU readings. For each task it keeps (a) the memory held by all its processes (the sum of their resident sets, from `ps`),
(b) the CPU cores it keeps busy (the change in the CPU time of its processes), (c) the utilisation and memory of the GPU it runs on. Notes:
  - GPU utilisation is a property of the GPU, not of one task: when light tasks share a GPU, every task on it gets the GPU's reading (the column `gpu_shared` says so);
  - the memory figure is the sum of resident sets of the task's processes, which counts shared pages more than once, so it can read a little above the true use; Slurm's own peak for the whole job
    is printed at the end of the job (sacct, in the sbatch script);
  - CPU time of processes that already exited is not seen between two samples, so `cores_avg` is a lower bound.
Outputs: one line per finished task in the log, a progress block every 10 minutes, `usage_summary.csv` (one row per finished task, rewritten after each), and a table by stage at the end.
"""
from __future__ import annotations

import csv
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

STAGE_RE = re.compile(r"^(?P<group>.*?)_(?P<stage>sam2|sam3|masks|clf_.+|ens|logits|select|track.*|score.*)$")
FIELDS = ["task", "group", "stage", "minutes", "gpu", "gpu_shared", "gpu_util_avg", "gpu_util_peak", "gpu_mem_peak_gb", "rss_avg_gb", "rss_peak_gb", "cores_avg", "cores_peak"]


def split_name(name: str) -> tuple[str, str]:
    m = STAGE_RE.match(name)
    if m:
        stage = m.group("stage")
        stage = re.sub(r"_sh\d+$", "", stage)           # tracking shards are one stage
        stage = re.sub(r"^clf_.*$", "classifier", stage)  # the four classifier members are one stage
        return m.group("group"), stage
    return name, name


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


class UsageMonitor:
    def __init__(self, out: Path) -> None:
        self.out = Path(out)
        self.t0 = time.time()
        self.live: dict[str, dict] = {}
        self.cpu_prev: dict[str, tuple[float, float]] = {}
        self.gpu_hist: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        self.rows: list[dict] = []
        self.total_rss = 0.0
        self.peak_total_rss = 0.0
        self.last_progress = 0.0

    # ----- sampling (every 15 s) -----
    def sample(self, running: dict[str, tuple[int, str | None]], gpu_latest: dict[str, tuple[float, float, float]]) -> None:
        now = time.time()
        for g, (u, used, total) in gpu_latest.items():
            self.gpu_hist[g] = [s for s in self.gpu_hist[g] if now - s[0] <= 900] + [(now, u, used, total)]
        try:
            res = subprocess.run(["ps", "-eo", "pgid=,rss=,cputimes="], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return
        per: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for line in res.stdout.splitlines():
            parts = line.split()
            if len(parts) == 3:
                try:
                    per[int(parts[0])][0] += float(parts[1]) / 1048576   # kB -> GB
                    per[int(parts[0])][1] += float(parts[2])
                except ValueError:
                    continue
        total = 0.0
        for name, (pgid, gpu) in running.items():
            rss, cpu = per.get(pgid, (0.0, 0.0))
            st = self.live.setdefault(name, {"rss": [], "cores": [], "util": [], "mem": [], "co": 1})
            st["rss"].append(rss)
            prev = self.cpu_prev.get(name)
            if prev and now > prev[1] and cpu >= prev[0]:
                st["cores"].append((cpu - prev[0]) / (now - prev[1]))
            self.cpu_prev[name] = (cpu, now)
            if gpu is not None and gpu in gpu_latest:
                u, used, _total = gpu_latest[gpu]
                st["util"].append(u)
                st["mem"].append(used / 1024)
                st["co"] = max(st["co"], sum(1 for _n, (_p, g) in running.items() if g == gpu))
            total += rss
        self.total_rss = total
        self.peak_total_rss = max(self.peak_total_rss, total)

    # ----- a task finished -----
    def task_done(self, name: str, gpu: str | None, wall_s: float) -> str:
        st = self.live.pop(name, None) or {"rss": [], "cores": [], "util": [], "mem": [], "co": 1}
        self.cpu_prev.pop(name, None)
        group, stage = split_name(name)
        row = {"task": name, "group": group, "stage": stage, "minutes": round(wall_s / 60, 1), "gpu": gpu if gpu is not None else "-", "gpu_shared": int(st["co"] > 1),
               "gpu_util_avg": round(mean(st["util"]), 0), "gpu_util_peak": round(max(st["util"], default=0), 0), "gpu_mem_peak_gb": round(max(st["mem"], default=0), 1),
               "rss_avg_gb": round(mean(st["rss"]), 1), "rss_peak_gb": round(max(st["rss"], default=0), 1),
               "cores_avg": round(mean(st["cores"]), 1), "cores_peak": round(max(st["cores"], default=0), 1)}
        self.rows.append(row)
        with open(self.out / "usage_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        gpu_txt = f"GPU {gpu}{' (shared)' if row['gpu_shared'] else ''} util avg {row['gpu_util_avg']:.0f}% peak {row['gpu_util_peak']:.0f}%, mem peak {row['gpu_mem_peak_gb']:.1f} GB | " if gpu is not None else ""
        return f"{gpu_txt}RSS avg {row['rss_avg_gb']:.1f} peak {row['rss_peak_gb']:.1f} GB | cores avg {row['cores_avg']:.1f} peak {row['cores_peak']:.1f}"

    # ----- progress block (every 10 minutes) -----
    def due(self, every_s: float = 600.0) -> bool:
        if time.time() - self.last_progress >= every_s:
            self.last_progress = time.time()
            return True
        return False

    def progress(self, groups: dict[str, tuple[int, int, list[str]]], running_info: list[tuple[str, str | None, float, float]], pending_minutes: float, n_gpus: int, done: int, total: int) -> str:
        """groups: group -> (done, total, running stage names); running_info: (task, gpu, minutes so far, estimated minutes)."""
        now = time.time()
        lines = [f"PROGRESS {done}/{total} tasks done, elapsed {(now - self.t0) / 3600:.1f} h"]
        for g, (d, t, run) in sorted(groups.items()):
            if d < t or run:
                lines.append(f"    {g:14s} {d:2d}/{t:2d} done" + (f", running: {', '.join(run)}" if run else ""))
        for name, gpu, mins, est in running_info:
            lines.append(f"    running {name:34s} gpu {gpu}  {mins:5.0f} of about {est:.0f} min")
        for g in sorted(self.gpu_hist):
            h = [s for s in self.gpu_hist[g] if now - s[0] <= 600]
            if h:
                lines.append(f"    GPU {g}: now {h[-1][1]:.0f}% util, {h[-1][2] / 1024:.1f}/{h[-1][3] / 1024:.0f} GB; last 10 min mean util {mean([s[1] for s in h]):.0f}%, peak {max(s[1] for s in h):.0f}%")
        lines.append(f"    memory held by the tasks' processes: {self.total_rss:.1f} GB now, {self.peak_total_rss:.1f} GB peak so far")
        ratio = [r["minutes"] for r in self.rows]
        if ratio and pending_minutes > 0:
            lines.append(f"    about {pending_minutes / max(1, n_gpus) / 60:.1f} h of estimated work left per GPU (the estimates are for an A100; the packing runs light tasks side by side)")
        return "\n".join(lines)

    # ----- end of the run -----
    def final_table(self) -> str:
        by: dict[str, list[dict]] = defaultdict(list)
        for r in self.rows:
            by[r["stage"]].append(r)
        lines = [f"{'stage':14s}{'n':>3s}{'min mean':>9s}{'GPU util avg':>13s}{'peak':>6s}{'GPU mem peak':>13s}{'RSS peak GB':>12s}{'cores avg':>10s}{'peak':>6s}"]
        for s, rs in sorted(by.items()):
            lines.append(f"{s:14s}{len(rs):3d}{mean([r['minutes'] for r in rs]):9.1f}{mean([r['gpu_util_avg'] for r in rs]):12.0f}%{max(r['gpu_util_peak'] for r in rs):5.0f}%"
                         f"{max(r['gpu_mem_peak_gb'] for r in rs):12.1f}G{max(r['rss_peak_gb'] for r in rs):12.1f}{mean([r['cores_avg'] for r in rs]):10.1f}{max(r['cores_peak'] for r in rs):6.1f}")
        gpu_min = sum(r["minutes"] for r in self.rows if r["gpu"] != "-")
        lines.append(f"GPU task-minutes {gpu_min:.0f} ({gpu_min / 60:.1f} h) in {(time.time() - self.t0) / 3600:.1f} h of wall time; peak memory held by all tasks together {self.peak_total_rss:.1f} GB")
        return "\n".join(lines)
