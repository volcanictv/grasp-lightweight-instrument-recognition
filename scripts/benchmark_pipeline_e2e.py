"""Whole-pipeline latency per keyframe, not component timing: every block is timed from the decoded frame to the final label, with the work it really does (decoding, box-prompted
segmentation, crops, the four classifier members, the evidential gate, tracking of the gated instruments over their window, re-classification of every tracked crop, evidential fusion).

Two modes, both CUDA-synchronised, on the keyframes of <bundle>/e2e_keyframes.json (test frames with their ground-truth boxes and the neighbouring frames of their windows):
    full   frame stage (decode, segmenter with flip average, crops, 4 members, gate S1 >= tau) and, in the same process and the same timer, the tracking of every gated instrument
           (tracker sam2 | yolo; none for no tracking). Writes masks and gate decisions for the track mode.
    track  tracking only, for the gated instruments of a previous full run (needed for EdgeTAM, which lives in its own environment and cannot build the SAM2.1 tiny segmenter).
With --all-instruments the tracking block is also timed for the instruments the gate did not select (after the keyframe's timer stops, so keyframe totals stay the real pipeline), which gives a stable
per-track cost from many more tracks than the gate selects.

Frame copying into the video-predictor folder is excluded, as in benchmark_tracker_latency.py; JPEG decoding of the window frames is included.

Usage (cluster job or titanxp, one idle GPU):
    python scripts/benchmark_pipeline_e2e.py --mode full --seg tiny --tracker sam2 --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml --sam2-checkpoint <large.pt> --causal --window 20 --out r/p4.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools import mask as mask_codec

from evaluate_temporal_track_ensemble import build_track_frame_nums, crop_from_box, crop_from_mask
from surgical_ai.data.region_dataset import GraspRegionDataset
from surgical_ai.data.transforms import build_transforms
from surgical_ai.evaluation.evidential import alpha_from_logits, variance_scores
from surgical_ai.models import build_model

ORDER = ["resnet50_320", "resnet50_224", "baseline", "letterbox_crop"]
WEIGHTS = {"resnet50_320": 0.40, "resnet50_224": 0.20, "baseline": 0.20, "letterbox_crop": 0.20}
SEG = {"tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"), "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
       "sam23": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml")}  # sam23: the final pipeline's masks, SAM2 large (bf16) and SAM3 (fp32), each flip-averaged, logits averaged


def sync() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def alpha_mix(logits: np.ndarray) -> np.ndarray:
    """logits (members in ORDER, classes) or (frames, members, classes) -> mixed Dirichlet parameters."""
    return sum(WEIGHTS[k] * alpha_from_logits(logits[..., ORDER.index(k), :]) for k in ORDER)


def fuse(det_frames: np.ndarray) -> int:
    """Frame-weighted evidential fusion of one track (frames, members, classes), the rule of scripts/gtbox_sam_final_eval.py."""
    alpha = alpha_mix(det_frames)
    mu = alpha / alpha.sum(axis=1, keepdims=True)
    return int(((mu.max(axis=1)[:, None]) * mu).sum(axis=0).argmax())


def summarize(v: list[float]) -> dict:
    if not v:
        return {"n": 0}
    s = sorted(v)
    return {"n": len(s), "mean_ms": statistics.mean(s), "median_ms": statistics.median(s), "p95_ms": s[int(0.95 * (len(s) - 1))], "max_ms": s[-1]}


class Classifier:
    def __init__(self, cfg_path: Path, dev: str, n_classes: int) -> None:
        cfg = yaml.safe_load(cfg_path.read_text())
        self.dev, self.members = dev, []
        for m in cfg["members"]:
            net = build_model(m["model"], num_classes=n_classes, pretrained=False, freeze_backbone=False).to(dev)
            net.load_state_dict(torch.load(REPO_ROOT / m["checkpoint"], map_location=dev), strict=False)
            self.members.append((m["label"], net.eval(), build_transforms(m["image_size"], train=False), m["letterbox"]))
        assert [m[0] for m in self.members] == ORDER, [m[0] for m in self.members]

    def logits(self, crops: dict[bool, np.ndarray | None]) -> np.ndarray | None:
        """Member logits (4, classes) of one frame's crops, keyed by letterbox flag."""
        if crops[True] is None:
            return None
        out = []
        with torch.no_grad():
            for _label, net, tf, letterbox in self.members:
                out.append(net(tf(Image.fromarray(crops[letterbox])).unsqueeze(0).to(self.dev)).float().cpu().numpy()[0])
        return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "track"], required=True)
    ap.add_argument("--seg", choices=["tiny", "large", "sam23"], default="tiny", help="full mode: the box-prompted segmenter (flip-averaged); sam23 = SAM2.1 large + SAM3")
    ap.add_argument("--tracker", choices=["none", "sam2", "yolo"], default="none", help="full mode; the track mode also takes the --sam2-config of EdgeTAM as sam2")
    ap.add_argument("--sam2-config")
    ap.add_argument("--sam2-checkpoint", type=Path)
    ap.add_argument("--yolo-weights", type=Path)
    ap.add_argument("--ckpt-dir", type=Path, default=Path(os.environ.get("SAM2_CKPT_DIR", Path.home() / "sam2" / "checkpoints")))
    ap.add_argument("--ensemble-config", type=Path, default=Path("configs/rc_ens4_N_s44.yaml"))
    ap.add_argument("--keyframes", type=Path, required=True, help="e2e_keyframes.json of the bundle")
    ap.add_argument("--gate-in", type=Path, help="track mode: the --out json of a full run (its masks and gate decisions are read from <out>.masks.pkl)")
    ap.add_argument("--sam2-weights", type=Path, help="fine-tuned SAM2 weights.pt (final_all8 for large, tiny_all8 for tiny), loaded over the base checkpoint like every benchmark of the pipeline")
    ap.add_argument("--sam3-weights", type=Path, help="fine-tuned SAM3 weights.pt (final_all8), loaded over the base model")
    ap.add_argument("--gate-saved", type=Path, help="json {instrument index: S1} of the paper's own run: the gate decision uses these scores (the base SAM2/SAM3 checkpoints here give other masks than the fine-tuned "
                    "ones, hence other S1), while the live S1 is still computed and recorded")
    ap.add_argument("--tau", type=float, default=2.85e-4, help="gate threshold on the epistemic score S1, fixed on fold 1")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2, help="leading keyframes run but not reported")
    ap.add_argument("--all-instruments", action="store_true")
    ap.add_argument("--sam3-random-init", action="store_true", help="dry runs only: SAM3 with random weights built from its config")
    ap.add_argument("--limit", type=int, default=0, help="first N keyframes only (dry runs)")
    ap.add_argument("--yolo-conf", type=float, default=0.1)
    ap.add_argument("--yolo-min-iou", type=float, default=0.1)
    ap.add_argument("--yolo-coast", type=int, default=2)
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("GRASP_DATA_ROOT", REPO_ROOT / "GraSP")))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = "cuda"
    if args.mode == "track" and args.tracker == "none":
        args.tracker = "sam2"

    ds = GraspRegionDataset(args.data_root, "test", letterbox=True)
    frames_root = args.data_root / "frames-001" / "frames"
    by_frame: dict[str, list[int]] = {}
    for i, inst in enumerate(ds.instances):
        by_frame.setdefault(inst[0], []).append(i)
    keyframes = json.loads(args.keyframes.read_text())["keyframes"]
    if args.limit:
        keyframes = keyframes[: args.limit]
    cls = Classifier(args.ensemble_config, dev, len(ds.class_names_ordered()))
    tmp = Path(os.environ.get("BENCH_TMP", "/tmp/bench_e2e_frames"))

    predictor = yolo = seg = sam3 = None
    if args.mode == "full":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from finetune_sam2_gtbox import box_logits, embed
        ck, cfg = SEG[args.seg]
        seg = SAM2ImagePredictor(build_sam2(cfg, str(args.ckpt_dir / ck), device=dev))
        if args.sam2_weights:
            print("sam2 weights:", seg.model.load_state_dict(torch.load(args.sam2_weights, map_location=dev), strict=False), flush=True)
        seg.model.eval()
        if args.seg == "sam23":
            import finetune_sam3_gtbox as s3
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
            if args.sam3_random_init:  # dry runs only (latency does not depend on the weights): the architecture from the config, no weight download
                from transformers import AutoConfig
                sam3 = Sam3TrackerModel(AutoConfig.from_pretrained("facebook/sam3")).to(dev).eval()
            else:
                sam3 = Sam3TrackerModel.from_pretrained("facebook/sam3").to(dev).eval()
            proc3 = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
            if args.sam3_weights:
                print("sam3 weights:", sam3.load_state_dict(torch.load(args.sam3_weights, map_location=dev), strict=False), flush=True)
                sam3.eval()
    if args.tracker == "sam2":
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(args.sam2_config, str(args.sam2_checkpoint), device=dev)
    elif args.tracker == "yolo":
        from ultralytics import YOLO
        from evaluate_temporal_track_yolo import encode, walk_track
        yolo = YOLO(str(args.yolo_weights))

    masks_in = gate_in = None
    if args.mode == "track":
        gate_in = json.loads(args.gate_in.read_text())
        masks_in = pickle.loads(Path(str(args.gate_in) + ".masks.pkl").read_bytes())
    saved_s1 = json.loads(args.gate_saved.read_text()) if args.gate_saved else None
    masks_out: dict[int, dict] = {}
    gate_out: dict[str, dict] = {}

    def window_of(file_name: str) -> tuple[list[int], int, str]:
        case, stem = file_name.split("/")
        nums, c = build_track_frame_nums(frames_root, case, int(stem.replace(".jpg", "")), args.window, 0 if args.causal else None)
        return nums, c, case

    def crops_of(frame: np.ndarray, mask: np.ndarray, box, center: bool) -> dict[bool, np.ndarray | None]:
        return {lb: (crop_from_box(frame, mask, box, lb) if center else crop_from_mask(frame, mask, lb)) for lb in (True, False)}

    def track_instrument(idx: int, mask: np.ndarray, det_by_frame: dict | None) -> dict:
        """Tracking of one instrument from the keyframe over its window, then the 4 members on every tracked crop and the evidential fusion. Timed by the caller."""
        file_name, _seg, box, _label = ds.instances[idx]
        nums, c, case = window_of(file_name)
        if args.tracker == "sam2":
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            for k, fn in enumerate(nums):
                shutil.copy(frames_root / case / f"{fn:05d}.jpg", tmp / f"{k:05d}.jpg")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            state = predictor.init_state(video_path=str(tmp))
            predictor.add_new_mask(state, frame_idx=c, obj_id=1, mask=mask)
            track = {c: mask}
            if not args.causal:
                for fi, _o, lg in predictor.propagate_in_video(state):
                    if fi != c:
                        track[fi] = (lg[0, 0] > 0).cpu().numpy()
            for fi, _o, lg in predictor.propagate_in_video(state, reverse=True):
                if fi != c:
                    track[fi] = (lg[0, 0] > 0).cpu().numpy()
            arrays = {li: np.array(Image.open(tmp / f"{li:05d}.jpg").convert("RGB")) for li in sorted(track)}
        else:
            t0 = time.perf_counter()
            per = det_by_frame
            ref = encode(mask)
            fwd = walk_track(ref, ref, [per[n] for n in nums[c + 1:]], args.yolo_min_iou, args.yolo_coast, False)
            bwd = walk_track(ref, ref, [per[n] for n in nums[:c][::-1]], args.yolo_min_iou, args.yolo_coast, False)
            track = {c: mask}
            track.update({c + 1 + i: mask_codec.decode(m).astype(bool) for i, m in fwd.items()})
            track.update({c - 1 - i: mask_codec.decode(m).astype(bool) for i, m in bwd.items()})
            arrays = {li: np.array(Image.open(frames_root / case / f"{nums[li]:05d}.jpg").convert("RGB")) for li in sorted(track)}
        frames = []
        for li in sorted(track):
            lg = cls.logits(crops_of(arrays[li], track[li], box, li == c))
            if lg is not None:
                frames.append(lg)
        pred = fuse(np.stack(frames))
        return {"track_len": len(frames), "pred": pred, "ms": 1000 * (sync() - t0)}

    def detect_window(file_name: str) -> dict:
        """YOLO26s-seg on every frame of the window in one batch (a live system keeps these detections, so this is the conservative cost), as {frame number: [RLE]}."""
        nums, _c, case = window_of(file_name)
        paths = [str(frames_root / case / f"{n:05d}.jpg") for n in nums]
        res = yolo.predict(paths, conf=args.yolo_conf, imgsz=640, retina_masks=True, device=dev, verbose=False, stream=False)
        return {n: ([] if r.masks is None else [encode(m) for m in r.masks.data.cpu().numpy().astype(bool)]) for n, r in zip(nums, res)}

    records, inst_records = [], []
    for k, file_name in enumerate(keyframes):
        indices = by_frame[file_name]
        boxes = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in (ds.instances[i][2] for i in indices)], dtype=np.float32)
        gated: dict[int, bool] = {}
        masks: dict[int, np.ndarray] = {}
        s1: dict[int, float] = {}
        rec: dict = {"keyframe": file_name, "n_instruments": len(indices)}
        t_kf = sync()
        if args.mode == "full":
            frame = np.array(Image.open(frames_root / file_name).convert("RGB"))
            flipped = frame[:, ::-1].copy()
            fboxes = boxes.copy()
            fboxes[:, [0, 2]] = frame.shape[1] - boxes[:, [2, 0]]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.seg == "sam23"):
                embed(seg, frame, grad=False)
                lg = box_logits(seg, boxes)[0].float()
                embed(seg, flipped, grad=False)
                lg = (lg + box_logits(seg, fboxes)[0].float().flip(-1)) / 2
            if sam3 is not None:
                with torch.no_grad():
                    image = Image.fromarray(frame)
                    a, _ = s3.predict(sam3, proc3, image, boxes, dev, grad=False)
                    b, _ = s3.predict(sam3, proc3, image.transpose(Image.FLIP_LEFT_RIGHT), fboxes, dev, grad=False)
                    lg = (lg + (a.float() + b.float().flip(-1)) / 2) / 2
            seg_masks = (lg[:, 0] > 0).cpu().numpy()
            for j, idx in enumerate(indices):
                if not seg_masks[j].any():
                    continue
                masks[idx] = seg_masks[j]
                member = cls.logits(crops_of(frame, masks[idx], ds.instances[idx][2], True))
                s1[idx] = float(variance_scores(alpha_mix(member)[None])["epistemic"][0])
                gated[idx] = (saved_s1.get(str(idx), s1[idx]) if saved_s1 else s1[idx]) >= args.tau
            rec["t_frame_ms"] = 1000 * (sync() - t_kf)
        else:
            for idx in indices:
                g = gate_in["gate"].get(str(idx))
                if g is not None:
                    masks[idx] = mask_codec.decode(masks_in[idx]).astype(bool)
                    s1[idx], gated[idx] = g["s1"], g["gated"]
            rec["t_frame_ms"] = gate_in["keyframes"][k]["t_frame_ms"]

        det = None
        order = [i for i in masks if gated[i]]
        if args.mode == "full" and args.tracker != "none" and order:
            if yolo is not None:
                t_det = sync()
                det = detect_window(file_name)
                rec["t_detect_ms"] = 1000 * (sync() - t_det)
            for idx in order:
                r = track_instrument(idx, masks[idx], det)
                inst_records.append({"index": idx, "keyframe": file_name, "gated": True, "s1": s1[idx], "track_ms": r["ms"], "track_len": r["track_len"], "label": r["pred"]})
            rec["t_track_ms"] = rec.get("t_detect_ms", 0.0) + sum(r["track_ms"] for r in inst_records[-len(order):])  # frame copying excluded, as in the propagation benchmark
            rec["t_total_ms"] = rec["t_frame_ms"] + rec["t_track_ms"]
        elif args.mode == "track":
            todo = [i for i in masks if gated[i] or args.all_instruments]
            if yolo is not None and todo:
                t_det = sync()
                det = detect_window(file_name)
                rec["t_detect_ms"] = 1000 * (sync() - t_det)
            for idx in todo:
                r = track_instrument(idx, masks[idx], det)
                inst_records.append({"index": idx, "keyframe": file_name, "gated": gated[idx], "s1": s1[idx], "track_ms": r["ms"], "track_len": r["track_len"], "label": r["pred"]})
        elif args.mode == "full":
            rec["t_track_ms"], rec["t_total_ms"] = 0.0, rec["t_frame_ms"]
        # untimed extra: the tracks of the instruments the gate did not select, for a stable per-track cost
        if args.mode == "full" and args.tracker != "none" and args.all_instruments:
            rest = [i for i in masks if not gated[i]]
            if yolo is not None and det is None and rest:
                det = detect_window(file_name)
            for idx in rest:
                r = track_instrument(idx, masks[idx], det)
                inst_records.append({"index": idx, "keyframe": file_name, "gated": False, "s1": s1[idx], "track_ms": r["ms"], "track_len": r["track_len"], "label": r["pred"]})
        rec["gated"] = int(sum(gated.values()))
        rec["instruments_gated"] = [i for i in masks if gated[i]]
        rec["warmup"] = k < args.warmup
        records.append(rec)
        for idx in masks:
            masks_out[idx] = mask_codec.encode(np.asfortranarray(masks[idx].astype(np.uint8)))
            gate_out[str(idx)] = {"s1": s1[idx], "gated": bool(gated[idx])}
        print(f"keyframe {k + 1}/{len(keyframes)} {file_name}: {len(indices)} instruments, {rec['gated']} gated, frame stage {rec['t_frame_ms']:.0f} ms"
              + (f", total {rec['t_total_ms']:.0f} ms" if "t_total_ms" in rec else ""), flush=True)

    kept = [r for r in records if not r["warmup"]]
    kept_idx = {r["keyframe"] for r in kept}
    tracks = [r for r in inst_records if r["keyframe"] in kept_idx]
    res = {"gpu": torch.cuda.get_device_name(0), "mode": args.mode, "seg": args.seg if args.mode == "full" else None, "tracker": args.tracker, "causal": args.causal, "window": args.window,
           "tau": args.tau, "keyframes": records, "instruments": {str(r["index"]): r for r in inst_records},
           "summary": {"keyframes": len(kept), "instruments": int(sum(r["n_instruments"] for r in kept)), "instruments_gated": int(sum(r["gated"] for r in kept)),
                       "frame_stage": summarize([r["t_frame_ms"] for r in kept]),
                       "keyframe_total": summarize([r["t_total_ms"] for r in kept if "t_total_ms" in r]),
                       "track_gated": summarize([r["track_ms"] for r in tracks if r["gated"]]),
                       "track_all": summarize([r["track_ms"] for r in tracks]),
                       "detect_per_keyframe": summarize([r["t_detect_ms"] for r in kept if "t_detect_ms" in r])}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "full":
        res["gate"] = gate_out
        args.out.with_name(args.out.name + ".masks.pkl").write_bytes(pickle.dumps(masks_out))
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
