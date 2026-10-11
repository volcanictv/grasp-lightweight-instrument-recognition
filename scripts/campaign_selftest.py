"""Tasks for testing scripts/campaign.py itself, with no models: short sleeps, one task that fails once and then works, one that always fails (its dependent must be skipped),
one CPU-only task, and one that is too long for the time budget.

    python scripts/campaign.py --out /tmp/cs_test --gpus 0,1 --budget-hours 0.02 --tasks-module campaign_selftest
"""
from __future__ import annotations


def build(cfg: dict, Task) -> list:
    out = cfg["out"]
    flaky = f"test -f {out}/flaky_ok || {{ touch {out}/flaky_ok; exit 3; }}; touch {out}/flaky.txt"
    return [
        Task("a", f"sleep 3; echo a > {out}/a.txt", outputs=[f"{out}/a.txt"], minutes=0.1),
        Task("b", f"sleep 3; echo b > {out}/b.txt", outputs=[f"{out}/b.txt"], minutes=0.1),
        Task("c_after_a_and_b", f"sleep 2; cat {out}/a.txt {out}/b.txt > {out}/c.txt; echo gpu=$CUDA_VISIBLE_DEVICES >> {out}/c.txt", deps=["a", "b"], outputs=[f"{out}/c.txt"], minutes=0.1),
        Task("flaky", flaky, outputs=[f"{out}/flaky.txt"], minutes=0.1),
        Task("always_fails", "exit 7", minutes=0.1, retries=0),
        Task("skipped_because_dep_failed", f"touch {out}/never.txt", deps=["always_fails"], outputs=[f"{out}/never.txt"], minutes=0.1),
        Task("cpu_only", f"sleep 2; touch {out}/cpu.txt", needs_gpu=False, outputs=[f"{out}/cpu.txt"], minutes=0.1),
        Task("too_long_for_the_budget", f"touch {out}/long.txt", outputs=[f"{out}/long.txt"], minutes=600),
    ]
