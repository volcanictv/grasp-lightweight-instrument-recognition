"""Merges the per-instrument IoU lists of scripts/eval_sam_ensemble.py (one file per GPU) into the fold1 table of the SAM3 protocol:
SAM2 (flip-averaged), SAM3 (flip-averaged), their ensemble, the registered adoption bar (0.9136) and the verdict. Also records the two SAM3
variants' own final fold1 checks. Usage: python scripts/merge_sam_ensemble.py --parts a.json b.json --variant enc4 --tta-dec ... --tta-enc4 ... --out ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAR = 0.9136  # SAM2 enc4 + flip average on fold1 (0.9086) plus the registered 0.005


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", type=Path, nargs="+", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tta-dec", type=Path, required=True)
    ap.add_argument("--tta-enc4", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows: dict[str, list] = {"sam2": [], "sam3": [], "ensemble": []}
    for p in args.parts:
        d = json.loads(p.read_text())
        for k in rows:
            rows[k].extend(d[k])
    res = {"fold": "fold1", "sam3_variant": args.variant, "bar": BAR, "instruments": len(rows["sam2"])}
    for k, v in rows.items():
        v = np.array(v)
        res[k] = {"mean_iou": float(v.mean()), "iou_ge_0.5": float((v >= 0.5).mean()), "iou_ge_0.75": float((v >= 0.75).mean())}
    res["sam3_final_check"] = {"dec": json.loads(args.tta_dec.read_text()), "enc4": json.loads(args.tta_enc4.read_text())}
    best = max(("sam2", "sam3", "ensemble"), key=lambda k: res[k]["mean_iou"])
    res["best"] = best
    sam3_ok = res["sam3"]["mean_iou"] >= BAR
    ens_ok = res["ensemble"]["mean_iou"] >= BAR and res["ensemble"]["mean_iou"] > max(res["sam2"]["mean_iou"], res["sam3"]["mean_iou"])
    res["verdict"] = ("adopt ensemble" if ens_ok and res["ensemble"]["mean_iou"] >= res["sam3"]["mean_iou"] else "adopt sam3" if sam3_ok else "adopt ensemble" if ens_ok else "keep sam2")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("sam2", "sam3", "ensemble", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
