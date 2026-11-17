"""GraSP two-fold validation: the released TAPIS fold models (own region proposals) with the ANNOTATED masks. Each annotated instrument takes the class distribution of the TAPIS prediction that overlaps its mask
most and is painted with its annotated mask; scored with TAPIS's own evaluator. usage (tapis_venv): python tapis_fold_gtmask.py"""
import json, os, sys
import numpy as np
from pathlib import Path
from pycocotools import mask as mc
sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox")); sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox/tapis"))
import tapis.evaluate.main_eval as ev

T = Path(os.path.expanduser("~/tapis_folds")); CL = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
fix = lambda s: {"size": list(s["size"]), "counts": s["counts"].decode() if isinstance(s["counts"], bytes) else s["counts"]}
dec = lambda r: mc.decode({"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}).astype(bool)
allm = {}
for f in (1, 2):
    gtp = str(T / "data/GraSP/annotations" / f"grasp_short-term_fold{f}.json"); g = json.load(open(gtp)); preds = json.load(open(T / "outputs" / f"fold{f}" / "epoch_0_preds_instruments.json"))
    res = {}; cov50 = 0; none = 0; acc = []
    for a in g["annotations"]:
        gr = fix(a["segmentation"]); gm = dec(gr); best, bi = None, -1.0
        for c in preds.get(a["image_name"], {"instances": []})["instances"]:
            pm = dec(fix(c["segment"])); inter = float((gm & pm).sum()); iou = inter / max(1.0, float(gm.sum() + pm.sum() - inter))
            if iou > bi: best, bi = c, iou
        cov50 += bi >= 0.5; none += (best is None or bi <= 0)
        dist = best["instruments_score_dist"] if best is not None and bi > 0 else [1 / 7.0] * 7
        acc.append(int(np.argmax(dist)) == int(a["category_id"]) - 1)
        x0, y0, w, h = [float(v) for v in a["bbox"]]
        res.setdefault(a["image_name"], {"instances": []})["instances"].append({"bbox": [x0, y0, x0 + w, y0 + h], "instruments_score_dist": dist, "segment": gr})
    for im in g["images"]: res.setdefault(im["file_name"], {"instances": []})
    n = len(g["annotations"]); print(f"fold{f}: annotated instruments {n}: matched at mask IoU>=0.5 for {cov50} ({100.0 * cov50 / n:.1f}%), no overlap {none}; class accuracy of the matched prediction {np.mean(acc):.4f}", flush=True)
    pj = T / "outputs" / f"fold{f}" / "annotated_masks_matched.json"; json.dump(res, open(pj, "w"))
    out = ev.main_per_task(gtp, str(pj), "instruments", "mIoU", masks_path=None); m = out[0] if isinstance(out, tuple) else out
    json.dump(m, open(T / "outputs" / f"fold{f}" / "annotated_masks_matched_metrics.json", "w"), indent=1); allm[f] = m
    print(f"fold{f}", {k: round(100 * v, 2) for k, v in m.items() if k in ("mIoU", "IoU", "mcIoU")}, flush=True)
mean = {k: float(np.mean([allm[f][k] for f in (1, 2)])) for k in allm[1] if isinstance(allm[1][k], float)}
json.dump(mean, open(T / "outputs" / "annotated_masks_matched_mean_metrics.json", "w"), indent=1)
print("MEAN of the two folds:", {k: round(100 * mean[k], 2) for k in ("mIoU", "IoU", "mcIoU")}); print("   per class IoU:", [round(100 * mean[c + "-IoU"], 1) for c in CL])
