"""SAM2 tracking for the instances the vote gate flags, on EndoVis 2018, using
the four members trained by endovis_pipeline_stage1.py. Same construction as on
GraSP: the annotated mask is propagated up to 10 frames each way, every frame
gets the weighted vote of 20 dropout passes per member, and frames are combined
by an uncertainty-weighted vote (a frame's weight is its own top vote share).

Usage (one shard per GPU):
    python scripts/endovis2018_track.py --zip ~/Desktop/endovis_data/endovis2018.zip \
        --stage1-dir experiments_endovis/stage1_2018 --shard-id 0 --num-shards 2 --device cuda:0
"""
from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image

import endovis_pipeline_stage1 as s1
import evaluate_endovis2018_generalization as e18
from analyze_uncertainty_signals import enable_mc_dropout
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model

NAME_RE = re.compile(r"(?P<dir>.*/val)/label/(?P<seq>seq_\d+)_frame(?P<n>\d+)\.bmp$")


def neighbours(names: set[str], m: re.Match, window: int) -> tuple[list[int], int]:
    def path(n: int) -> str:
        return f"{m['dir']}/image/{m['seq']}_frame{n:03d}.bmp"
    center = int(m["n"])
    nums = [center]
    for k in range(1, window + 1):
        if path(center - k) not in names:
            break
        nums.insert(0, center - k)
    center_idx = len(nums) - 1
    for k in range(1, window + 1):
        if path(center + k) not in names:
            break
        nums.append(center + k)
    return nums, center_idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--stage1-dir", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path.home() / "sam2/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--tmp-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many instances")
    args = ap.parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tmp_dir = args.tmp_dir or Path(f"/tmp/endovis2018_track_{args.shard_id}")
    zf = zipfile.ZipFile(args.zip)
    names = set(zf.namelist())
    instances = e18.extract_instances(zf, "val")
    votes = np.load(args.stage1_dir / "votes.npz")
    flagged = np.where(votes["u"] >= s1.GATE)[0]
    mine = flagged[args.shard_id::args.num_shards]
    if args.limit:
        mine = mine[:args.limit]
    print(f"{len(flagged)} flagged of {len(instances)}; this shard {len(mine)}", flush=True)

    members = []
    for cfg in s1.MEMBERS:
        model = build_model(cfg["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(args.stage1_dir / f"{cfg['label']}.pt", map_location=device))
        model.eval()
        members.append((cfg, model, build_transforms(cfg["size"], train=False)))
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    def frame_share(crops: dict[bool, np.ndarray]) -> np.ndarray:
        s = np.zeros(7)
        for cfg, model, tf in members:
            x = tf(Image.fromarray(crops[cfg["letterbox"]])).unsqueeze(0).to(device)
            with torch.no_grad():
                enable_mc_dropout(model)
                logits = model(x.repeat(s1.MC, 1, 1, 1))
                model.eval()
            s += cfg["weight"] * (logits.argmax(1).cpu().numpy()[:, None] == np.arange(7)).mean(axis=0)
        return s

    out_json = args.stage1_dir / f"tracked_shard{args.shard_id}.json"
    out_npz = args.stage1_dir / f"tracked_shard{args.shard_id}.npz"
    results, shares = [], {}
    for n, idx in enumerate(mine):
        image, gt_mask, bbox, label_name, src = instances[idx]
        m = NAME_RE.match(src)
        nums, center_idx = neighbours(names, m, args.window)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        frames = []
        for li, fn in enumerate(nums):
            arr = np.array(Image.open(io.BytesIO(zf.read(f"{m['dir']}/image/{m['seq']}_frame{fn:03d}.bmp"))).convert("RGB"))
            frames.append(arr)
            Image.fromarray(arr).save(tmp_dir / f"{li:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=str(tmp_dir))
        predictor.add_new_mask(state, frame_idx=center_idx, obj_id=1, mask=gt_mask)
        masks = {center_idx: gt_mask}
        for reverse in (False, True):
            for fi, _o, ml in predictor.propagate_in_video(state, reverse=reverse):
                if fi != center_idx:
                    masks[fi] = (ml[0, 0] > 0).cpu().numpy()
        frame_shares = []
        for li in sorted(masks):
            mk = masks[li]
            if li != center_idx and mk.sum() < e18.MIN_COMPONENT_AREA:
                continue
            ys, xs = np.nonzero(mk)
            box = bbox if li == center_idx else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            crops = {lb: e18.crop_instance(frames[li], mk, box, lb) for lb in (True, False)}
            frame_shares.append(frame_share(crops))
        fs = np.stack(frame_shares)
        weights = fs.max(axis=1)
        weighted = (weights[:, None] * fs).sum(axis=0)
        pooled = fs.sum(axis=0)
        true = int(votes["y"][idx])
        results.append({"index": int(idx), "true": true, "track_len": len(fs),
                        "weighted_pred": int(weighted.argmax()), "pooled_pred": int(pooled.argmax()),
                        "single_pred": int(votes["pred"][idx])})
        shares[str(idx)] = fs
        print(f"[{n + 1}/{len(mine)}] idx {idx} true {true} len {len(fs)} weighted {results[-1]['weighted_pred']}", flush=True)
        if (n + 1) % args.save_every == 0 or n + 1 == len(mine):
            out_json.write_text(json.dumps(results))
            np.savez_compressed(out_npz, **shares)


if __name__ == "__main__":
    main()
