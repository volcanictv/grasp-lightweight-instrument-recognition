"""SAM2 temporal tracking once, frame classification by several ensembles at the same time.

Same mechanism as evaluate_temporal_track_ensemble.py (propagate the ground-truth mask +-window
frames with SAM2, crop every frame), but each frame is classified by every ensemble given in
--ensemble-configs (for example the evidential official-split ensembles of seeds 42, 43 and 44),
and only the deterministic per-member logits are saved, one array per ensemble and instance:
`det_<tag>_<index>` with shape (frames, members, classes). The frame list and the centre position
are stored once per instance (`frames_<index>`, `center_<index>`). Instances come from
--indices-json (a JSON list, or a dict with "indices") in the order given, sharded by
--shard-id/--num-shards, so a priority order survives an early stop. Progress is checkpointed
every --save-every instances and the run is resumable (finished instances are skipped).

Usage:
    python scripts/evaluate_temporal_track_multi.py --indices-json idx.json \
        --ensemble-configs s42=configs/evidential/ens_E_official_s42_lam0p01a10.yaml \
                           s43=configs/evidential/ens_E_official_s43_lam0p01a10.yaml \
        --sam2-checkpoint ~/sam2/checkpoints/sam2.1_hiera_large.pt \
        --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --shard-id 0 --num-shards 2 \
        --device cuda:0 --frame-logits-out out_shard0.npz
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image

from evaluate_temporal_track_ensemble import build_track_frame_nums, crop_from_box, crop_from_mask
from surgical_ai.data.mask_utils import decode_instance_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.models import build_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-configs", nargs="+", required=True, help="tag=path pairs")
    ap.add_argument("--indices-json", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--sam2-checkpoint", type=Path, required=True)
    ap.add_argument("--sam2-config", type=str, required=True)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--tmp-dir", type=Path, default=Path("/tmp/sam2_track_multi"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frame-logits-out", type=Path, required=True)
    return ap.parse_args()


def load_store(path: Path) -> dict:
    if not path.exists():
        return {}
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main() -> None:
    args = parse_args()
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(args.device)
    frames_root = args.data_root / "frames-001" / "frames"
    ds = GraspRegionDataset(args.data_root, args.split, letterbox=True)
    n_classes = len(ds.class_names_ordered())

    ensembles = {}
    for item in args.ensemble_configs:
        tag, path = item.split("=", 1)
        cfg = yaml.safe_load(Path(path).read_text())
        models, transforms = [], []
        for m in cfg["members"]:
            model = build_model(m["model"], num_classes=n_classes, pretrained=False, freeze_backbone=False).to(device)
            model.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=device), strict=False)
            model.eval()
            models.append(model)
            transforms.append((build_transforms(m["image_size"], train=False), m["letterbox"]))
        ensembles[tag] = (models, transforms, [m["label"] for m in cfg["members"]])
        print(f"loaded ensemble {tag}: {[m['label'] for m in cfg['members']]}", flush=True)

    raw = json.loads(args.indices_json.read_text())
    order = raw["indices"] if isinstance(raw, dict) else raw
    indices = order[args.shard_id :: args.num_shards]
    store = load_store(args.frame_logits_out)
    done = {int(k[len("frames_"):]) for k in store if k.startswith("frames_")}
    print(f"shard {args.shard_id}/{args.num_shards}: {len(indices)} instances, {len(done)} already saved", flush=True)

    predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=str(device))

    def flush() -> None:
        args.frame_logits_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.frame_logits_out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **store)
        os.replace(tmp, args.frame_logits_out)

    for n, idx in enumerate(indices):
        if idx in done:
            continue
        file_name, segmentation, box, label = ds.instances[idx]
        case, frame_stem = file_name.split("/")
        center_num = int(frame_stem.replace(".jpg", ""))
        gt_mask = decode_instance_mask(segmentation).astype(bool)
        frame_nums, center_idx = build_track_frame_nums(frames_root, case, center_num, args.window)

        tmp_dir = args.tmp_dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        for local_idx, fn in enumerate(frame_nums):
            shutil.copy(frames_root / case / f"{fn:05d}.jpg", tmp_dir / f"{local_idx:05d}.jpg")

        state = predictor.init_state(video_path=str(tmp_dir))
        predictor.add_new_mask(state, frame_idx=center_idx, obj_id=1, mask=gt_mask)
        track_masks: dict[int, np.ndarray] = {center_idx: gt_mask}
        for frame_idx, _o, mask_logits in predictor.propagate_in_video(state):
            if frame_idx != center_idx:
                track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()
        for frame_idx, _o, mask_logits in predictor.propagate_in_video(state, reverse=True):
            if frame_idx != center_idx:
                track_masks[frame_idx] = (mask_logits[0, 0] > 0).cpu().numpy()

        sorted_local = sorted(track_masks)
        frame_arrays = {li: np.array(Image.open(tmp_dir / f"{li:05d}.jpg").convert("RGB")) for li in sorted_local}
        crop_cache: dict[tuple[int, bool], np.ndarray | None] = {}

        def get_crop(li: int, letterbox: bool):
            key = (li, letterbox)
            if key not in crop_cache:
                if li == center_idx:
                    crop_cache[key] = crop_from_box(frame_arrays[li], track_masks[li], box, letterbox)
                else:
                    crop_cache[key] = crop_from_mask(frame_arrays[li], track_masks[li], letterbox)
            return crop_cache[key]

        valid = [li for li in sorted_local if get_crop(li, True) is not None]
        if not valid:
            print(f"[{n}] instance {idx}: no usable track frames, skipping", flush=True)
            continue
        per_tag = {}
        for tag, (models, transforms, _labels) in ensembles.items():
            frames_out = []
            for li in valid:
                row = []
                for model, (transform, letterbox) in zip(models, transforms):
                    crop = get_crop(li, letterbox)
                    image = transform(Image.fromarray(crop)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        row.append(model(image).float().cpu().numpy()[0])
                frames_out.append(np.stack(row))
            per_tag[tag] = np.stack(frames_out)
        for tag, arr in per_tag.items():
            store[f"det_{tag}_{idx}"] = arr
        store[f"frames_{idx}"] = np.array(valid)
        store[f"center_{idx}"] = np.array(valid.index(center_idx) if center_idx in valid else 0)
        done.add(idx)
        print(f"[{len(done)}] {case}/{frame_stem} instance {idx}: {len(valid)} frames", flush=True)
        if len(done) % args.save_every == 0:
            flush()
    flush()
    print("finished shard", args.shard_id, flush=True)


if __name__ == "__main__":
    main()
