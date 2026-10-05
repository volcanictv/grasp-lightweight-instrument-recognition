"""Reduces a fine-tuned SAM checkpoint (scripts/finetune_sam2_gtbox.py, scripts/finetune_sam3_gtbox.py) to the tensors that differ from the
base model, for the public package: the fine-tuning saved whole blocks (868 MB for SAM2, 1.83 GB for SAM3) of which only the decoder, prompt
encoder, neck and the last four encoder blocks actually changed. Base weights plus this delta, loaded with strict=False, reproduce the
fine-tuned model exactly (checked here by comparing every tensor).

Usage (titanxp, sam3_venv):
    python scripts/make_sam_delta.py --model sam2 --weights experiments/sam2_gtbox/final_all8/weights.pt --out release/sam2_delta.pt
    python scripts/make_sam_delta.py --model sam3 --weights experiments/sam3/final_all8/weights.pt --out release/sam3_delta.pt
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["sam2", "sam3"], required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sam2-checkpoint", type=Path, default=Path.home() / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt")
    args = ap.parse_args()

    tuned = torch.load(args.weights, map_location="cpu")
    if args.model == "sam2":
        base = torch.load(args.sam2_checkpoint, map_location="cpu")["model"]
    else:
        from transformers import Sam3TrackerModel
        base = Sam3TrackerModel.from_pretrained("facebook/sam3").state_dict()

    delta, same, missing = {}, 0, []
    for key, value in tuned.items():
        if key not in base:
            missing.append(key)
            delta[key] = value.clone()
        elif base[key].shape != value.shape or not torch.equal(base[key].to(value.dtype), value):
            delta[key] = value.clone()
        else:
            same += 1
    print(f"{args.model}: {len(tuned)} saved tensors, {len(delta)} differ from the base, {same} identical, {len(missing)} not in the base")
    if missing:
        print("not in base (kept):", missing[:5])
    # round trip: base overlaid with the delta must equal the full fine-tuned tensors
    for key, value in tuned.items():
        merged = delta.get(key, base[key] if key in base else None)
        assert merged is not None and torch.equal(merged.to(value.dtype), value), f"round trip differs at {key}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, args.out)
    size = args.out.stat().st_size
    print(f"wrote {args.out}: {size / 1e6:.1f} MB (from {args.weights.stat().st_size / 1e6:.1f} MB), sha256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
