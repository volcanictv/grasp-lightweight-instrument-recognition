"""EndoVis 2018 (ISINet split): the retrained TAPIS (own proposals) with the ANNOTATED masks. Each annotated instrument takes the class distribution of the TAPIS prediction that overlaps its mask most
(mask IoU); the instrument is painted with its annotated mask; scored with TAPIS's own evaluator. Also scores the raw predictions as a check (must reproduce the published-style row).
usage (tapis_venv): python tapis_endovis_gtmask.py DIR   (DIR holds final_1gpu_epoch_0_preds_instruments.json, val_anns.json, val_anns_rle.json)"""
import json, os, sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mc
sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox")); sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox/tapis"))
import tapis.evaluate.main_eval as ev

D = Path(sys.argv[1]); gtp = str(D / "val_anns_rle.json"); g = json.load(open(gtp)); preds = json.load(open(D / "final_1gpu_epoch_0_preds_instruments.json"))
fix = lambda s: {"size": list(s["size"]), "counts": s["counts"].decode() if isinstance(s["counts"], bytes) else s["counts"]}


def run(pred_dict, name):
    pj = D / f"{name}.json"; json.dump(pred_dict, open(pj, "w"))
    out = ev.main_per_task(gtp, str(pj), "instruments", "mIoU", masks_path=None); m = out[0] if isinstance(out, tuple) else out
    json.dump(m, open(D / f"{name}_metrics.json", "w"), indent=1)
    print(name, {k: round(100 * v, 2) for k, v in m.items() if k in ("mIoU", "IoU", "mcIoU")}); print("   per class:", {k.replace("-IoU", ""): round(100 * v, 1) for k, v in m.items() if k.endswith("-IoU")})


run(preds, "check_raw_predictions")
res = {}; cov50 = 0; cov10 = 0; none = 0; acc = []
for a in g["annotations"]:
    gr = fix(a["rle"] if isinstance(a.get("rle"), dict) else a["segmentation"]); gm = mc.decode({"size": gr["size"], "counts": gr["counts"].encode()}) if isinstance(gr["counts"], str) else mc.decode(gr)
    cands = preds[a["image_name"]]["instances"]; best, bi = None, -1.0
    for c in cands:
        pr = fix(c["segment"]); pm = mc.decode({"size": pr["size"], "counts": pr["counts"].encode()}); inter = float((gm.astype(bool) & pm.astype(bool)).sum()); iou = inter / max(1.0, float(gm.sum() + pm.sum() - inter))
        if iou > bi: best, bi = c, iou
    cov50 += bi >= 0.5; cov10 += bi >= 0.1
    dist = best["instruments_score_dist"] if best is not None and bi > 0 else [1 / 7] * 7; none += (best is None or bi <= 0)
    acc.append(int(np.argmax(dist)) == int(a["category_id"]) - 1)
    x0, y0, w, h = [float(v) for v in a["bbox"]]
    res.setdefault(a["image_name"], {"instances": []})["instances"].append({"bbox": [x0, y0, x0 + w, y0 + h], "instruments_score_dist": dist, "segment": gr})
for im in g["images"]: res.setdefault(im["file_name"], {"instances": []})
n = len(g["annotations"])
print(f"annotated instruments {n}: best TAPIS prediction overlaps at mask IoU>=0.5 for {cov50} ({100*cov50/n:.1f}%), >=0.1 for {cov10}, no overlap at all {none}; class accuracy of the matched prediction {np.mean(acc):.4f}")
run(res, "annotated_masks_matched")
