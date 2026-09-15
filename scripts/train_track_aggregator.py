"""Learns a per-frame reliability weight for combining a SAM2-propagated
track's per-frame predictions, instead of the fixed uniform average
(`docs/DECISIONS.md` 2026-09-15's avg-softmax baseline). Generalizes that
rule: avg-softmax is the special case where every frame gets weight 1/N
regardless of how reliable it actually looks.

Requires the richer JSON produced by the current
`evaluate_temporal_track_classification.py` (each instance has a "frames"
list with offset/area_frac/softmax per track frame) -- the earlier
fold1_full.json/fold2_full.json from the plain pilot do NOT have this and
need to be regenerated first.

Tiny model, tiny data (a few hundred to a thousand tracks): a 3-input MLP
(area_frac, normalized frame offset, that frame's own max-softmax
confidence) maps each frame to a raw score; scores are softmax-normalized
across the track into weights that sum to 1; the weighted sum of each
frame's softmax becomes the track-level prediction. Trained with negative
log-likelihood on the true label directly, no separate "does aggregation
help" heuristic -- the network is optimized for exactly the accuracy
metric being reported.

Cross-fold discipline matches the rest of this project: train the
aggregator on one fold's tracks, evaluate on the other's, in both
directions, and report both -- not just whichever direction looks best.

Usage:
    python scripts/train_track_aggregator.py \\
        --fold1-json docs/reports/temporal_track_fold1_full.json \\
        --fold2-json docs/reports/temporal_track_fold2_full.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold1-json", type=Path, required=True)
    parser.add_argument("--fold2-json", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/reports"))
    return parser.parse_args()


class FrameWeightNet(nn.Module):
    """area_frac, normalized offset, frame confidence -> raw reliability score."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def load_tracks(path: Path, window: int) -> list[dict]:
    data = json.loads(path.read_text())
    tracks = []
    for instance in data["instances"]:
        if "frames" not in instance:
            raise ValueError(
                f"{path} has no per-frame data -- regenerate with the current "
                "evaluate_temporal_track_classification.py, the earlier pilot JSON predates it"
            )
        features = torch.tensor(
            [[f["area_frac"], f["offset"] / window, max(f["softmax"])] for f in instance["frames"]],
            dtype=torch.float32,
        )
        softmax = torch.tensor([f["softmax"] for f in instance["frames"]], dtype=torch.float32)
        tracks.append({"features": features, "softmax": softmax, "true_idx": instance["true_idx"]})
    return tracks


def track_probs(model: FrameWeightNet, track: dict) -> torch.Tensor:
    scores = model(track["features"])
    weights = torch.softmax(scores, dim=0)
    return (weights.unsqueeze(1) * track["softmax"]).sum(dim=0)


def train_and_eval(train_tracks: list[dict], eval_tracks: list[dict], epochs: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    model = FrameWeightNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for track in train_tracks:
            probs = track_probs(model, track)
            loss = loss + -torch.log(probs[track["true_idx"]].clamp_min(1e-8))
        loss = loss / len(train_tracks)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = sum(
            int(track_probs(model, track).argmax().item() == track["true_idx"]) for track in eval_tracks
        )
    return {"n": len(eval_tracks), "learned_weight_accuracy": correct / len(eval_tracks), "final_train_loss": float(loss.item())}


def uniform_baseline(tracks: list[dict]) -> float:
    correct = sum(int(track["softmax"].mean(dim=0).argmax().item() == track["true_idx"]) for track in tracks)
    return correct / len(tracks)


def main() -> None:
    args = parse_args()
    fold1 = load_tracks(args.fold1_json, args.window)
    fold2 = load_tracks(args.fold2_json, args.window)
    print(f"fold1: {len(fold1)} tracks, fold2: {len(fold2)} tracks")

    results = {}
    for train_name, train_tracks, eval_name, eval_tracks in [
        ("fold1", fold1, "fold2", fold2),
        ("fold2", fold2, "fold1", fold1),
    ]:
        eval_uniform_acc = uniform_baseline(eval_tracks)
        outcome = train_and_eval(train_tracks, eval_tracks, args.epochs, args.lr, args.seed)
        delta = outcome["learned_weight_accuracy"] - eval_uniform_acc
        results[f"train_{train_name}_eval_{eval_name}"] = {
            "eval_uniform_avg_softmax_accuracy": eval_uniform_acc,
            "eval_learned_weight_accuracy": outcome["learned_weight_accuracy"],
            "delta": delta,
            "final_train_loss": outcome["final_train_loss"],
        }
        print(f"train={train_name} eval={eval_name}: uniform={eval_uniform_acc:.4f} "
              f"learned={outcome['learned_weight_accuracy']:.4f} delta={delta:+.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "track_aggregator_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
