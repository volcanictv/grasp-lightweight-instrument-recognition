"""SAM2 tracking of EndoVis 2018 instances flagged by the evidential gate, using
the evidential members trained by endovis_evidential_stage1.py (docs/DECISIONS.md
2026-09-29). The annotated mask is propagated up to 10 frames each way; every
kept frame gets ONE deterministic pass per member (no MC passes), and the
per-frame member logits are saved so the pre-registered evidential frame
combine (evidential_gate_final_eval.evidential_track_predict) can be applied
offline. Instance indices come from a json list (the union of gates A and B).

Usage (one shard per GPU):
    python scripts/endovis2018_track_evidential.py --zip ~/Desktop/endovis_data/endovis2018.zip \
        --members-dir experiments_endovis/evid_stage1_2018 --indices experiments_endovis/evid_stage1_2018/flagged_union.json \
        --shard-id 0 --num-shards 2 --device cuda:0
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

import endovis_evidential_stage1 as es1
import evaluate_endovis2018_generalization as e18
from endovis2018_track import NAME_RE, neighbours
from evidential_gate_final_eval import evidential_track_predict
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--members-dir", type=Path, required=True)
    ap.add_argument("--indices", type=Path, required=True)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path.home() / "sam2/checkpoints/sam2.1_hiera_large.pt")
    ap.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--tmp-dir", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after this many instances")
    args = ap.parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(args.device)
    tmp_dir = args.tmp_dir or Path(f"/tmp/endovis2018_track_evid_{args.shard_id}")
    zf = zipfile.ZipFile(args.zip)
    names = set(zf.namelist())
    instances = e18.extract_instances(zf, "val")
    flagged = np.array(json.loads(args.indices.read_text())["indices"])
    mine = flagged[args.shard_id::args.num_shards]
    if args.limit:
        mine = mine[:args.limit]
    print(f"{len(flagged)} flagged of {len(instances)}; this shard {len(mine)}", flush=True)

    members = []
    for cfg in es1.MEMBERS:
        model = build_model(cfg["model"], num_classes=7, pretrained=False, freeze_backbone=False).to(device)
        model.load_state_dict(torch.load(args.members_dir / f"{cfg['label']}.pt", map_location=device))
        model.eval()
        members.append((cfg, model, build_transforms(cfg["size"], train=False)))
    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    def frame_logits(crops: dict[bool, np.ndarray]) -> np.ndarray:
        rows = []
        for cfg, model, tf in members:
            x = tf(Image.fromarray(crops[cfg["letterbox"]])).unsqueeze(0).to(device)
            with torch.no_grad():
                rows.append(model(x).float().cpu().numpy()[0])
        return np.stack(rows)

    out_json = args.members_dir / f"tracked_shard{args.shard_id}.json"
    out_npz = args.members_dir / f"frames_shard{args.shard_id}.npz"
    results, frames_store = [], {}
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
        per_frame = []
        for li in sorted(masks):
            mk = masks[li]
            if li != center_idx and mk.sum() < e18.MIN_COMPONENT_AREA:
                continue
            ys, xs = np.nonzero(mk)
            box = bbox if li == center_idx else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            crops = {lb: e18.crop_instance(frames[li], mk, box, lb) for lb in (True, False)}
            per_frame.append(frame_logits(crops))
        det = np.stack(per_frame)
        frames_store[f"det_{idx}"] = det
        results.append({"index": int(idx), "track_len": int(len(det)), "pred": evidential_track_predict(det)})
        print(f"[{n + 1}/{len(mine)}] idx {idx} len {len(det)} pred {results[-1]['pred']}", flush=True)
        if (n + 1) % args.save_every == 0 or n + 1 == len(mine):
            out_json.write_text(json.dumps(results))
            np.savez_compressed(out_npz, **frames_store)


if __name__ == "__main__":
    main()
