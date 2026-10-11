#!/usr/bin/env python3
"""A small task scheduler for one Slurm allocation: the "packed job" pattern of the RC documentation (one allocation, many independent tasks), written out as a plain Python loop.

What it does, in order:
  1. loads a task list (scripts/campaign_tasks.py builds it; every task is a shell command with dependencies and expected output files)
  2. starts a task on each free GPU (one task per GPU, with CUDA_VISIBLE_DEVICES set to that GPU, so every command simply uses cuda:0) and CPU-only tasks in a few spare slots
  3. writes a done marker when a task's command exits 0 and its output files exist; a rerun after a cancel, a time limit or a crash skips every finished task
  4. never starts a task whose estimate does not fit in the time left, and stops launching new tasks near the deadline
  5. logs GPU utilisation every minute (RC may cancel jobs with idle GPUs) and warns when a GPU has been idle for ten minutes
It does not touch anything outside --out except what the tasks' own commands write. On SIGTERM (Slurm's time limit) it stops its children and exits; their tasks have no done marker and rerun.

Usage:
    python scripts/campaign.py --out ~/grasp_work/campaign --gpus 0,1 --budget-hours 27 [--smoke] [--only NAME,NAME] [--list]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign_usage import UsageMonitor, split_name  # noqa: E402


@dataclass
class Task:
    name: str
    cmd: str                                   # a shell command; it sees one GPU as cuda:0
    needs_gpu: bool = True
    deps: list[str] = field(default_factory=list)
    minutes: float = 10.0                      # estimate: used by the time-budget guard and the plan
    outputs: list[str] = field(default_factory=list)   # files that must exist after the command succeeds
    priority: int = 0                          # higher first among the tasks that are ready
    retries: int = 1                           # extra attempts after a failure
    light: bool = False                        # small GPU footprint and mostly waiting on the CPU: up to --slots-per-gpu light tasks may share one GPU (never with a heavy task)
    mem_gb: float = 8.0                        # GPU memory the task needs at most: a light task is added to a busy GPU only when that much (plus a margin) is free


class Campaign:
    def __init__(self, tasks: list[Task], out: Path, gpus: list[str], cpu_slots: int, deadline: float, slots_per_gpu: int = 1, max_share: int = 3, util_target: float = 50.0) -> None:
        self.slots_per_gpu = max(1, slots_per_gpu)
        self.max_share, self.util_target = max(self.slots_per_gpu, max_share), util_target
        self.samples: dict[str, list[tuple[float, float, float, float]]] = {g: [] for g in gpus}   # per GPU: (time, utilisation %, memory used MiB, memory total MiB)
        self.last_sample = 0.0
        self.last_start: dict[str, float] = {}
        self.usage = UsageMonitor(out)
        self.tasks = {t.name: t for t in tasks}
        assert len(self.tasks) == len(tasks), "duplicate task names"
        for t in tasks:
            assert all(d in self.tasks for d in t.deps), f"{t.name}: unknown dependency"
        self.out, self.gpus, self.cpu_slots, self.deadline = out, gpus, cpu_slots, deadline
        for sub in ("done", "logs"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        self.running: dict[str, tuple[subprocess.Popen, str | None, float]] = {}   # name -> (process, gpu, start)
        self.failed: set[str] = set()
        self.attempts: dict[str, int] = {}
        self.stop = False
        self.idle_since: dict[str, float] = {}
        self.last_util = 0.0

    # ----- state -----
    def marker(self, t: Task) -> Path:
        return self.out / "done" / f"{t.name}.json"

    def done(self, t: Task) -> bool:
        return self.marker(t).exists()

    def blocked(self, t: Task) -> bool:
        return any(d in self.failed or self.blocked(self.tasks[d]) for d in t.deps)

    def ready(self) -> list[Task]:
        out = [t for t in self.tasks.values() if not self.done(t) and t.name not in self.running and t.name not in self.failed and not self.blocked(t)
               and all(self.done(self.tasks[d]) for d in t.deps)]
        return sorted(out, key=lambda t: (-t.priority, t.name))

    # ----- running -----
    def start(self, t: Task, gpu: str | None) -> None:
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        log = open(self.out / "logs" / f"{t.name}.log", "a")
        log.write(f"\n[{time.strftime('%F %T')}] start (attempt {self.attempts.get(t.name, 0) + 1}), gpu {gpu}\n$ {t.cmd}\n")
        log.flush()
        proc = subprocess.Popen(["bash", "-c", t.cmd], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.running[t.name] = (proc, gpu, time.time())
        if gpu is not None:
            self.last_start[gpu] = time.time()
        print(f"[{time.strftime('%T')}] start  {t.name:42s} gpu {gpu}  (about {t.minutes:.0f} min)", flush=True)

    def sample(self) -> None:
        """Every 15 s: utilisation and memory of each GPU (nvidia-smi), kept for the last five minutes; schedule() uses them to decide whether a GPU can take one more light task."""
        if time.time() - self.last_sample < 15:
            return
        self.last_sample = time.time()
        latest: dict[str, tuple[float, float, float]] = {}
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            lines = res.stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            lines = []
        for line in lines:
            try:
                idx, util, used, total = [x.strip() for x in line.split(",")]
                if idx in self.samples:
                    self.samples[idx] = (self.samples[idx] + [(time.time(), float(util), float(used), float(total))])[-20:]
                    latest[idx] = (float(util), float(used), float(total))
            except ValueError:
                continue
        # the same readings, plus the processes of every running task, go to the usage monitor (scripts/campaign_usage.py)
        self.usage.sample({n: (p.pid, g) for n, (p, g, _t) in self.running.items()}, latest)

    def progress_block(self) -> str:
        groups: dict[str, list] = {}
        for t in self.tasks.values():
            grp, stage = split_name(t.name)
            d = groups.setdefault(grp, [0, 0, []])
            d[1] += 1
            d[0] += int(self.done(t))
            if t.name in self.running:
                d[2].append(stage)
        info = [(n, g, (time.time() - t0) / 60, self.tasks[n].minutes) for n, (_p, g, t0) in self.running.items()]
        pending = sum(t.minutes for t in self.tasks.values() if t.needs_gpu and not self.done(t))
        return self.usage.progress({k: (v[0], v[1], v[2]) for k, v in groups.items()}, info, pending, len(self.gpus), sum(self.done(t) for t in self.tasks.values()), len(self.tasks))

    def underused(self, gpu: str, need_gb: float) -> bool:
        """True when this GPU's mean utilisation over the last two minutes is below the target and it has the memory (plus a 3 GB margin) for another task of need_gb."""
        recent = [s for s in self.samples.get(gpu, []) if time.time() - s[0] <= 120]
        if len(recent) < 4 or time.time() - self.last_start.get(gpu, 0.0) < 120:   # a task that has just started is still loading: do not judge yet
            return False
        mean_util = sum(s[1] for s in recent) / len(recent)
        free_gb = (recent[-1][3] - recent[-1][2]) / 1024
        return mean_util < self.util_target and free_gb >= need_gb + 3.0

    def finish(self, name: str, rc: int) -> None:
        proc, gpu, t0 = self.running.pop(name)
        t = self.tasks[name]
        missing = [o for o in t.outputs if not Path(o).exists()]
        minutes = (time.time() - t0) / 60
        usage_txt = self.usage.task_done(name, gpu, minutes * 60)
        if rc == 0 and not missing:
            self.marker(t).write_text(json.dumps({"minutes": round(minutes, 1), "gpu": gpu, "finished": time.strftime("%F %T"), "cmd": t.cmd, "usage": usage_txt}))
            print(f"[{time.strftime('%T')}] done   {name:42s} {minutes:6.1f} min | {usage_txt}", flush=True)
            return
        self.attempts[name] = self.attempts.get(name, 0) + 1
        why = f"exit {rc}" if rc != 0 else f"missing outputs {missing[:2]}"
        if self.attempts[name] <= t.retries and not self.stop:
            print(f"[{time.strftime('%T')}] RETRY  {name} ({why})", flush=True)
        else:
            self.failed.add(name)
            print(f"[{time.strftime('%T')}] FAILED {name} ({why}); its dependents will be skipped. See {self.out / 'logs' / (name + '.log')}", flush=True)

    def schedule(self) -> None:
        load: dict[str, list[bool]] = {g: [] for g in self.gpus}   # per GPU: the light flags of its running tasks
        for n, (_p, g, _t) in self.running.items():
            if g is not None and g in load:
                load[g].append(self.tasks[n].light)
        n_cpu = sum(1 for n, (_p, g, _t) in self.running.items() if g is None)
        remaining = self.deadline - time.time()
        for t in self.ready():
            if self.stop or t.minutes * 60 * 1.15 > remaining:
                continue
            if t.needs_gpu:
                if t.light:   # an EMPTY GPU first (spread the work: every GPU should be busy), and only when none is empty stack onto a GPU of light tasks
                    # a GPU of light tasks takes another one while it has a free slot, or (up to --max-share) while it is measured below the utilisation target and has the memory
                    fits = [g for g in self.gpus if not load[g]] or \
                        [g for g in self.gpus if load[g] and all(load[g]) and (len(load[g]) < self.slots_per_gpu or (len(load[g]) < self.max_share and self.underused(g, t.mem_gb)))]
                else:
                    fits = [g for g in self.gpus if not load[g]]
                if fits:
                    self.start(t, fits[0])
                    load[fits[0]].append(t.light)
            elif n_cpu < self.cpu_slots:
                self.start(t, None)
                n_cpu += 1

    def gpu_log(self) -> None:
        if time.time() - self.last_util < 60:
            return
        self.last_util = time.time()
        try:
            res = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return
        busy_gpus = {g for _p, g, _t in self.running.values() if g is not None}
        with open(self.out / "gpu_util.csv", "a") as f:
            for line in res.stdout.strip().splitlines():
                idx, util, mem = [x.strip() for x in line.split(",")]
                f.write(f"{time.strftime('%F %T')},{idx},{util},{mem},{'task' if idx in busy_gpus else 'no task'}\n")
        for g in self.gpus:
            if g in busy_gpus:
                self.idle_since.pop(g, None)
            else:
                since = self.idle_since.setdefault(g, time.time())
                if time.time() - since > 600:
                    print(f"[{time.strftime('%T')}] WARNING GPU {g} has had no task for {(time.time() - since) / 60:.0f} minutes", flush=True)

    def status(self) -> None:
        pend = [t.name for t in self.tasks.values() if not self.done(t) and t.name not in self.running and t.name not in self.failed]
        (self.out / "status.json").write_text(json.dumps({"time": time.strftime("%F %T"), "running": sorted(self.running), "failed": sorted(self.failed),
                                                          "done": sum(self.done(t) for t in self.tasks.values()), "pending": pend, "total": len(self.tasks)}, indent=1))

    def run(self) -> int:
        def on_term(_sig, _frm):
            self.stop = True
            for proc, _g, _t in self.running.values():
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGINT, on_term)
        while True:
            for name in list(self.running):
                rc = self.running[name][0].poll()
                if rc is not None:
                    self.finish(name, rc)
            if self.stop and not self.running:
                print("stopped on request; finished tasks are kept, the rest rerun on resubmission", flush=True)
                break
            self.sample()
            if not self.stop:
                self.schedule()
            self.gpu_log()
            self.status()
            if self.usage.due():
                print(f"[{time.strftime('%T')}] " + self.progress_block(), flush=True)
            startable = [t for t in self.ready() if t.minutes * 60 * 1.15 <= self.deadline - time.time()]
            if not self.running and not startable:
                break  # everything is done, failed, blocked, or too long for the time that is left
            time.sleep(2)
        skipped = [t.name for t in self.tasks.values() if not self.done(t)]
        summary = {t.name: ("done" if self.done(t) else "failed" if t.name in self.failed else "blocked" if self.blocked(t) else "not started") for t in self.tasks.values()}
        (self.out / "campaign_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"finished: {sum(v == 'done' for v in summary.values())} of {len(summary)} tasks done; not done: {skipped}", flush=True)
        table = self.usage.final_table()
        print(table, flush=True)
        (self.out / "usage_final.txt").write_text(table + "\n")
        return 0 if not skipped else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="GPU ids to use, comma separated (default: the ones Slurm gave this job)")
    ap.add_argument("--budget-hours", type=float, default=24.0, help="no task starts that would not finish inside this many hours from now")
    ap.add_argument("--cpu-slots", type=int, default=2, help="CPU-only tasks that may run next to the GPU tasks")
    ap.add_argument("--slots-per-gpu", type=int, default=1, help="light tasks (classifier members, logits) that may share one GPU; heavy tasks (training, tracking) always get a whole GPU")
    ap.add_argument("--max-share", type=int, default=3, help="most light tasks on one GPU when it is measured below --util-target and has the memory")
    ap.add_argument("--util-target", type=float, default=50.0, help="a GPU whose utilisation (mean of the last two minutes) is below this percentage may take one more light task")
    ap.add_argument("--smoke", action="store_true", help="tiny versions of every task, for testing the whole chain")
    ap.add_argument("--tasks-module", default="campaign_tasks")
    ap.add_argument("--only", default="", help="comma separated task names (their dependencies are included)")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {"out": str(args.out.resolve()), "smoke": args.smoke, "code": str(Path(__file__).resolve().parents[1])}
    tasks: list[Task] = importlib.import_module(args.tasks_module).build(cfg, Task)
    if args.only:
        want, by = set(), {t.name: t for t in tasks}
        def add(n):
            if n not in want:
                want.add(n)
                for d in by[n].deps:
                    add(d)
        for n in args.only.split(","):
            add(n)
        tasks = [t for t in tasks if t.name in want]
    if args.list:
        total = sum(t.minutes for t in tasks if t.needs_gpu) / 60
        for t in tasks:
            print(f"{t.name:44s} {'gpu' if t.needs_gpu else 'cpu'}  {t.minutes:7.0f} min  after: {', '.join(t.deps) or '-'}")
        print(f"{len(tasks)} tasks, about {total:.1f} GPU-hours, about {total / max(1, len(args.gpus.split(','))):.1f} hours on {len(args.gpus.split(','))} GPUs if the graph keeps them busy")
        return 0
    c = Campaign(tasks, args.out, [g.strip() for g in args.gpus.split(",") if g.strip() != ""], args.cpu_slots, time.time() + args.budget_hours * 3600, args.slots_per_gpu, args.max_share, args.util_target)
    return c.run()


if __name__ == "__main__":
    sys.exit(main())
