"""Softmax, MC-dropout and deep-ensemble baselines on the TEST crops, next to the evidential score, on the same instruments (the final SAM2+SAM3 masks of scripts/gtbox_sam_final_eval's registered run).

Members: the cross-entropy deep-dropout members of arm C trained on the official training cases with the evidential finals' recipe (scripts/make_ce_official_configs.py, scripts/ce_baseline_tasks.py; the test cases
are never read in training). Evidential: the arm N finals (logits_s<seed>.npz). Methods and signals as in scripts/calibration_baselines.py: softmax ensemble (max-softmax, entropy, energy), MC dropout (20 passes per
member, 80 in all: mean-softmax, vote disagreement, entropy, mutual information), deep ensemble (3 seeds, 12 networks: max-softmax, entropy, mutual information), evidential (epistemic S1, max-belief, entropy),
evidential pooled over the 3 seeds. Metrics: accuracy, ECE (15 bins), NLL, AUROC for flagging errors, share of all errors found when the 20% highest scores are reviewed. Paired bootstrap of the evidential score S1
against every other signal (difference in AUROC and in errors found), resampling instruments and resampling the five cases.

    python scripts/baselines_on_test.py extract --ce-dir ~/ce_campaign          # GPU: writes experiments/gtbox_sam/final/ce_test_s<seed>.npz
    python scripts/baselines_on_test.py analyze --out docs/reports/gtbox_sam/baselines_on_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from scipy.special import logsumexp, softmax

from calibration_baselines import calib, detect, entropy
from evidential_analyze import LABELS, WEIGHTS
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores

SEEDS = (42, 43, 44)
D = REPO_ROOT / "experiments" / "gtbox_sam" / "final"
MC_PASSES = 20


def extract(args: argparse.Namespace) -> None:
    import torch
    import yaml
    from PIL import Image
    from pycocotools import mask as mask_codec

    from evaluate_temporal_track_ensemble import crop_from_box
    from surgical_ai.data.region_dataset import GraspRegionDataset
    from surgical_ai.data.transforms import build_transforms
    from surgical_ai.models import build_model

    dev = "cuda"
    data_root = Path(os.environ["GRASP_DATA_ROOT"])
    ds = GraspRegionDataset(data_root, "test", letterbox=True)
    frames_root = data_root / "frames-001" / "frames"
    masks = pickle.loads((D / "masks.pkl").read_bytes())
    idx = sorted(masks)
    if args.limit:
        idx = idx[: args.limit]
    members = {}
    for s in SEEDS:
        for m in LABELS:
            cfg = yaml.safe_load((REPO_ROOT / f"configs/arms/armC_{m}_official_s{s}.yaml").read_text())
            run = Path((Path(args.ce_dir) / f"{m}_s{s}.ok").read_text().strip())
            run = run if run.is_absolute() else REPO_ROOT / run
            net = build_model(cfg["model"]["name"], num_classes=7, pretrained=False, freeze_backbone=False).to(dev)
            net.load_state_dict(torch.load(run / "best.pt", map_location=dev), strict=False)
            size, lb = cfg["data"]["image_size"], bool(cfg["data"].get("letterbox_crop", False))
            members[(s, m)] = (net.eval(), size, lb)
    tf_by_key = {}
    for (_s, _m), (_net, size, lb) in members.items():
        tf_by_key.setdefault((size, lb), build_transforms(size, train=False))

    def dropout_on(net) -> None:
        for mod in net.modules():
            if isinstance(mod, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                mod.train()

    det = {k: [] for k in members}
    mc = {k: [] for k in members}
    cache: dict[str, np.ndarray] = {}
    t0, B = time.time(), 32
    for start in range(0, len(idx), B):
        chunk = idx[start:start + B]
        tens: dict[tuple, list] = {}
        for i in chunk:
            file_name, _seg, box, _label = ds.instances[i]
            case, stem = file_name.split("/")
            key = str(frames_root / case / stem)
            if key not in cache:
                if len(cache) > 40:
                    cache.clear()
                cache[key] = np.array(Image.open(key).convert("RGB"))
            mask = mask_codec.decode(masks[i]).astype(bool)
            crops = {lb: crop_from_box(cache[key], mask, box, lb) for lb in (True, False)}
            for k, tf in tf_by_key.items():
                tens.setdefault(k, []).append(tf(Image.fromarray(crops[k[1]])))
        batch = {k: torch.stack(v).to(dev) for k, v in tens.items()}
        with torch.no_grad():
            for (s, m), (net, size, lb) in members.items():
                x = batch[(size, lb)]
                net.eval()
                det[(s, m)].append(net(x).float().cpu().numpy())
                dropout_on(net)
                mc[(s, m)].append(np.stack([net(x).float().cpu().numpy() for _ in range(MC_PASSES)]))   # (passes, B, 7)
                net.eval()
        if (start // B) % 10 == 0:
            print(f"{start + len(chunk)}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
    for s in SEEDS:
        payload = {"index": np.array(idx)}
        for m in LABELS:
            payload[f"det_{m}"] = np.concatenate(det[(s, m)], axis=0)
            payload[f"mc_{m}"] = np.concatenate(mc[(s, m)], axis=1)
        np.savez_compressed(D / f"ce_test_s{s}.npz", **payload)
        print("wrote", D / f"ce_test_s{s}.npz")


def auroc_and_caught(score: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    r = detect(score, err)
    return r["auroc"], r["caught20"]


def analyze(args: argparse.Namespace) -> None:
    from surgical_ai.data.region_dataset import GraspRegionDataset

    data_root = Path(os.environ["GRASP_DATA_ROOT"])
    ds = GraspRegionDataset(data_root, "test", letterbox=True)
    ce = {s: np.load(D / f"ce_test_s{s}.npz") for s in SEEDS}
    idx = ce[42]["index"]
    ev = {s: np.load(D / f"logits_s{s}.npz") for s in SEEDS}
    y = ev[42]["y"][idx]
    cases = np.array([ds.instances[i][0].split("/")[0] for i in idx])
    sigs: dict[str, dict[int, dict]] = {}   # method -> seed -> {"p": probs, "pred": predictions, "scores": {name: score}}

    for s in SEEDS:
        logits = [ce[s]["det_" + m] for m in LABELS]
        p_soft = sum(w * softmax(z, axis=1) for w, z in zip(WEIGHTS, logits))
        zbar = sum(w * z for w, z in zip(WEIGHTS, logits))
        sigs.setdefault("Softmax ensemble", {})[s] = {"p": p_soft, "scores": {"max-softmax": 1 - p_soft.max(1), "entropy": entropy(p_soft), "energy": -logsumexp(zbar, axis=1)}}
        mc = [softmax(ce[s]["mc_" + m], axis=2) for m in LABELS]   # each (20, N, 7)
        p_mc = sum(w * x.mean(axis=0) for w, x in zip(WEIGHTS, mc))
        mean_h = sum(w * entropy(x).mean(axis=0) for w, x in zip(WEIGHTS, mc))
        votes = np.concatenate([x.argmax(axis=2) for x in mc], axis=0)   # (80, N)
        counts = np.stack([(votes == c).sum(axis=0) for c in range(7)], axis=1)   # (N, 7)
        vote_pred, vote_share = counts.argmax(axis=1), counts.max(axis=1) / votes.shape[0]
        sigs.setdefault("MC dropout (80 passes)", {})[s] = {"p": p_mc, "pred_override": {"vote disagreement": vote_pred},
                                                            "scores": {"vote disagreement": 1 - vote_share, "max-softmax": 1 - p_mc.max(1), "entropy": entropy(p_mc), "mutual information": entropy(p_mc) - mean_h}}
        alpha = sum(w * alpha_from_logits(ev[s]["det_" + m][idx]) for w, m in zip(WEIGHTS, LABELS))
        mu = alpha / alpha.sum(axis=1, keepdims=True)
        sigs.setdefault("Evidential (1 pass)", {})[s] = {"p": mu, "scores": {"epistemic S1": variance_scores(alpha)["epistemic"], "max-belief": 1 - mu.max(1), "entropy": entropy(mu)}}
    P = [sigs["Softmax ensemble"][s]["p"] for s in SEEDS]
    p_de = sum(P) / 3
    pooled = {"Deep ensemble (3 seeds, 12 nets)": {"p": p_de, "scores": {"max-softmax": 1 - p_de.max(1), "entropy": entropy(p_de), "mutual information": entropy(p_de) - sum(entropy(p) for p in P) / 3}}}
    alpha_pool = sum(sum(w * alpha_from_logits(ev[s]["det_" + m][idx]) for w, m in zip(WEIGHTS, LABELS)) for s in SEEDS) / 3
    mu_pool = alpha_pool / alpha_pool.sum(axis=1, keepdims=True)
    pooled["Evidential, 3 seeds (12 nets)"] = {"p": mu_pool, "scores": {"epistemic S1": variance_scores(alpha_pool)["epistemic"], "max-belief": 1 - mu_pool.max(1), "entropy": entropy(mu_pool)}}
    nets = {"Softmax ensemble": 4, "MC dropout (80 passes)": 80, "Evidential (1 pass)": 4, "Deep ensemble (3 seeds, 12 nets)": 12, "Evidential, 3 seeds (12 nets)": 12}

    res: dict = {"instruments": int(len(idx)), "cases": {c: int((cases == c).sum()) for c in sorted(set(cases))}, "methods": {}}
    for name, entries in {**sigs, **{k: {0: v} for k, v in pooled.items()}}.items():
        cal = [calib(v["p"], y) for v in entries.values()]
        entry = {"networks": nets[name], "seeds": len(entries), "calibration": {k: float(np.mean([c[k] for c in cal])) for k in ("accuracy", "nll", "brier", "ece")}, "signals": {}}
        for sig in next(iter(entries.values()))["scores"]:
            vals = [auroc_and_caught(v["scores"][sig], v.get("pred_override", {}).get(sig, v["p"].argmax(1)) != y) for v in entries.values()]
            entry["signals"][sig] = {"auroc": float(np.mean([a for a, _ in vals])), "caught20": float(np.mean([c for _, c in vals])), "caught20_per_seed": [float(c) for _, c in vals], "auroc_per_seed": [float(a) for a, _ in vals]}
        res["methods"][name] = entry

    # paired bootstrap: evidential S1 (seed-mean statistics) against every other signal, resampling instruments and resampling cases
    rng = np.random.default_rng(0)
    ucases = sorted(set(cases))
    by_case = {c: np.where(cases == c)[0] for c in ucases}
    def stats(sel: np.ndarray, items: list[tuple[dict, str]]) -> np.ndarray:
        out = []
        for entries, sig in items:
            a, c = [], []
            for v in entries.values():
                pred = v.get("pred_override", {}).get(sig, v["p"].argmax(1))
                err = (pred != y)[sel]
                if err.sum() == 0 or err.sum() == len(err):
                    a.append(np.nan); c.append(np.nan)
                    continue
                r = detect(v["scores"][sig][sel], err)
                a.append(r["auroc"]); c.append(r["caught20"])
            out.append([np.nanmean(a), np.nanmean(c)])
        return np.array(out)   # (signals, 2)
    all_methods = {**sigs, **{k: {0: v} for k, v in pooled.items()}}
    items = [(sigs["Evidential (1 pass)"], "epistemic S1")]   # the reference: the evidential score of the single 4-network ensemble, mean over the seeds
    labels = []
    for name, entries in all_methods.items():
        for sig in next(iter(entries.values()))["scores"]:
            if name == "Evidential (1 pass)" and sig == "epistemic S1":
                continue
            labels.append(f"{name}: {sig}")
            items.append((entries, sig))
    point = stats(np.arange(len(y)), items)
    B_inst, B_case = 1000, 2000
    inst_draws = np.array([stats(rng.integers(0, len(y), len(y)), items) for _ in range(B_inst)])
    case_draws = np.array([stats(np.concatenate([by_case[ucases[j]] for j in rng.integers(0, len(ucases), len(ucases))]), items) for _ in range(B_case)])
    res["paired_vs_epistemic_S1"] = {}
    for k, lab in enumerate(labels, start=1):
        d_inst, d_case = inst_draws[:, 0, :] - inst_draws[:, k, :], case_draws[:, 0, :] - case_draws[:, k, :]
        res["paired_vs_epistemic_S1"][lab] = {m: {"point": float(point[0, j] - point[k, j]), "ci95_instruments": [float(np.nanpercentile(d_inst[:, j], 2.5)), float(np.nanpercentile(d_inst[:, j], 97.5))],
                                                  "ci95_cases": [float(np.nanpercentile(d_case[:, j], 2.5)), float(np.nanpercentile(d_case[:, j], 97.5))]} for j, m in enumerate(("auroc", "caught20"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"{'method / signal':52s}{'acc':>7s}{'ECE':>7s}{'NLL':>7s}{'AUROC':>8s}{'caught@20':>10s}")
    for name, e in res["methods"].items():
        for sig, v in e["signals"].items():
            print(f"{name + ': ' + sig:52s}{e['calibration']['accuracy']:7.3f}{e['calibration']['ece']:7.3f}{e['calibration']['nll']:7.3f}{v['auroc']:8.3f}{100 * v['caught20']:9.1f}%")
    print("S1 minus other, caught@20 (points) with 95% CI over instruments / over cases")
    for lab, v in res["paired_vs_epistemic_S1"].items():
        c = v["caught20"]
        print(f"  vs {lab:50s}{100 * c['point']:+6.1f}  [{100 * c['ci95_instruments'][0]:+.1f}, {100 * c['ci95_instruments'][1]:+.1f}]  cases [{100 * c['ci95_cases'][0]:+.1f}, {100 * c['ci95_cases'][1]:+.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--ce-dir", required=True)
    e.add_argument("--limit", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else analyze(args)


if __name__ == "__main__":
    main()
