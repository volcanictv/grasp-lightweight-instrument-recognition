"""Test of the utilisation-aware packing of scripts/campaign.py: eight light tasks that sleep (the GPUs stay idle, so utilisation is far below 50%). With 1 slot per GPU two start at once;
after two minutes of low utilisation each GPU may take more, up to --max-share 3. A heavy task (not light) must never share."""
def build(cfg, Task):
    out = cfg["out"]
    tasks = [Task(f"light{i}", f"sleep 200; touch {out}/l{i}", outputs=[f"{out}/l{i}"], minutes=4, light=True, mem_gb=1, priority=9 - i) for i in range(8)]
    tasks.append(Task("heavy", f"sleep 20; touch {out}/h", outputs=[f"{out}/h"], minutes=1, priority=-5))
    return tasks
