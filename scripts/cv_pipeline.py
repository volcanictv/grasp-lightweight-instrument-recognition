"""Small helpers for the per-fold tasks of the campaign (scripts/campaign_tasks.py). Each subcommand does one step that has no script of its own.

  clf-run      write the config of one classifier member of a fold and train it with scripts/train.py (the recipe of the official arm N members: evidential loss, lambda 0.01 annealed over 10 epochs,
               20 epochs, fixed schedule with the last epoch kept, tracker-style crops with probability 0.5), then write <dir>/<member>.ok with the checkpoint path
  ens-config   write the four-member ensemble config of a fold from those markers
  select       choose the instruments to track: the highest epistemic scores S1 of a fold's single-pass logits (a fraction, or all, or at most N)
  score        run scripts/gtbox_sam_final_eval.py on a fold with budgets given as fractions of its instruments

A fold is named by its registry prefix, e.g. cv5_f0 or cv5_smoke (see scripts/build_cv_splits.py); GRASP_EXTRA_SPLITS must name the registry file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import yaml

import evidential_make_configs as mk

MEMBER_ORDER = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]


def cmd_clf_run(a: argparse.Namespace) -> None:
    cfg = mk.train_config("E", a.member, a.fold, a.seed, 0.01, 10)
    cfg["training"]["select_best"] = False  # fixed schedule: the last epoch is kept, nothing is selected
    cfg["training"]["epochs"] = a.epochs
    cfg["data"]["val_split_override"] = f"{a.fold}_dev"  # inside the training data: the held-out cases are never read
    dirs = [str(Path(d)) for d in a.neighbours if (Path(d) / "meta.json").exists()]
    if dirs:
        cfg["data"]["neighbour_dir"] = dirs[0] if len(dirs) == 1 else dirs
        cfg["data"]["neighbour_prob"] = 0.5
    stem = f"cvclf_{a.fold}_{a.member}_s{a.seed}"
    d = Path(a.dir)
    (d / "configs").mkdir(parents=True, exist_ok=True)
    cfg_path = d / "configs" / f"{stem}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run([a.python, "scripts/train.py", str(cfg_path), "--data-root", a.data_root, "--device", "cuda:0", "--num-workers", str(a.workers),
                    "--experiments-dir", str(d / "runs")], cwd=REPO_ROOT, check=True)
    runs = sorted(p for p in (d / "runs").glob(f"{stem}_2*") if (p / "best.pt").exists())
    if not runs:
        raise SystemExit(f"no finished run for {stem}")
    (d / f"{a.member}.ok").write_text(json.dumps({"checkpoint": str(runs[-1] / "best.pt"), "member": a.member, "neighbour_dirs": dirs}))
    print("trained", stem, runs[-1])


def cmd_ens_config(a: argparse.Namespace) -> None:
    members = []
    for label in MEMBER_ORDER:
        info = json.loads((Path(a.dir) / f"{label}.ok").read_text())
        m = mk.MEMBERS[label]
        members.append({"checkpoint": info["checkpoint"], "model": m["e"], "image_size": m["size"], "letterbox": m["letterbox"], "label": label})
    Path(a.out).write_text(yaml.safe_dump({"weight_resnet50_320": mk.WEIGHT_320, "members": members}, sort_keys=False))
    print("wrote", a.out)


def cmd_select(a: argparse.Namespace) -> None:
    from evidential_seeds_e2e_eval import CONFIGS, alpha_mix
    from surgical_ai.evaluation.evidential import variance_scores

    z = np.load(a.logits)
    s1 = variance_scores(alpha_mix(lambda k: z["det_" + k], CONFIGS["four"]))["epistemic"]
    done = np.where(z["done"])[0]
    order = done[np.argsort(-s1[done], kind="stable")]
    n = len(order) if a.frac >= 1.0 else int(round(a.frac * len(order)))
    if a.max_instruments:
        n = min(n, a.max_instruments)
    chosen = sorted(int(i) for i in order[:n])
    Path(a.out).write_text(json.dumps({"errors": [{"index": i} for i in chosen]}))
    for k in range(a.shards if a.shards > 1 else 0):   # interleaved by S1 rank, so the shards cost about the same
        by_rank = sorted(chosen, key=lambda i: -s1[i])
        Path(a.out).with_name(f"{Path(a.out).stem}_shard{k}.json").write_text(json.dumps({"errors": [{"index": i} for i in sorted(by_rank[k :: a.shards])]}))
    print(f"tracking {len(chosen)} of {len(done)} instruments (the highest S1), in {max(1, a.shards)} shard(s)")


def cmd_score(a: argparse.Namespace) -> None:
    z = np.load(a.logits)
    n = int(z["done"].sum())
    budgets = sorted({max(1, int(round(f * n))) for f in a.fracs})
    cmd = [a.python, "scripts/gtbox_sam_final_eval.py", "--masks", a.masks, "--logits", a.logits, "--tracked-dir", a.tracked_dir, "--split", a.split, "--budgets", *map(str, budgets),
           "--save-frames", a.frames_out, "--out", a.out]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clf-run")
    c.add_argument("--fold", required=True), c.add_argument("--member", required=True, choices=MEMBER_ORDER), c.add_argument("--seed", type=int, default=42)
    c.add_argument("--dir", required=True), c.add_argument("--neighbours", nargs="*", default=[]), c.add_argument("--data-root", required=True)
    c.add_argument("--epochs", type=int, default=20), c.add_argument("--workers", type=int, default=4), c.add_argument("--python", default=sys.executable)
    e = sub.add_parser("ens-config")
    e.add_argument("--dir", required=True), e.add_argument("--out", required=True)
    s = sub.add_parser("select")
    s.add_argument("--logits", required=True), s.add_argument("--frac", type=float, default=1.0), s.add_argument("--max-instruments", type=int, default=0), s.add_argument("--shards", type=int, default=1), s.add_argument("--out", required=True)
    r = sub.add_parser("score")
    r.add_argument("--split", required=True), r.add_argument("--masks", required=True), r.add_argument("--logits", required=True), r.add_argument("--tracked-dir", required=True)
    r.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.2, 0.29, 0.4, 0.5]), r.add_argument("--frames-out", required=True), r.add_argument("--out", required=True)
    r.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    {"clf-run": cmd_clf_run, "ens-config": cmd_ens_config, "select": cmd_select, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
