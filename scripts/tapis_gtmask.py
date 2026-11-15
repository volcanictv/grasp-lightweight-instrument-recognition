"""TAPIS (released model, ground-truth boxes) with the ANNOTATED masks: the class distribution TAPIS's second stage assigns to each given box is kept, and the instrument is painted with its
annotated mask instead of the matched Mask2Former segment; scored with TAPIS's own evaluator. Run once with TAPIS's segments (must reproduce the given-box row) and once with the annotated masks.
usage (tapis_venv): python tapis_gtmask.py OUT_PREFIX"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox")); sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox/tapis"))
import tapis.evaluate.main_eval as ev

T = Path(os.path.expanduser("~/tapis_gt")); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
gtp = str(T / "annotations" / "grasp_short-term_test.json"); g = json.load(open(gtp)); preds = json.load(open(T / "outputs" / "gt_mode" / "epoch_0_preds_instruments.json"))
CL = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors", "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


res = {"tapis_segments": {}, "annotated_masks": {}}; hit = 0; weak = 0; correct = 0; y = []
for a in g["annotations"]:
    x0, y0, w, h = [float(v) for v in a["bbox"]]; box = [x0, y0, x0 + w, y0 + h]
    cands = preds[a["image_name"]]["instances"]
    best = max(cands, key=lambda c: iou(box, c["bbox"])); s = iou(box, best["bbox"]); hit += s > 0.95; weak += s <= 0.95
    seg = a["segmentation"]; gseg = {"size": list(seg["size"]), "counts": seg["counts"].decode() if isinstance(seg["counts"], bytes) else seg["counts"]}
    pseg = best["segment"]; pseg = {"size": list(pseg["size"]), "counts": pseg["counts"].decode() if isinstance(pseg["counts"], bytes) else pseg["counts"]}
    for key, sg in (("tapis_segments", pseg), ("annotated_masks", gseg)):
        res[key].setdefault(a["image_name"], {"instances": []})["instances"].append({"bbox": box, "instruments_score_dist": best["instruments_score_dist"], "segment": sg})
    correct += int(np.argmax(best["instruments_score_dist"]) == int(a["category_id"]) - 1); y.append(int(a["category_id"]) - 1)
for key in res:
    for im in g["images"]: res[key].setdefault(im["file_name"], {"instances": []})
print(f"annotations {len(g['annotations'])}: matched to a prediction with box IoU > 0.95: {hit}, weaker: {weak}; instance accuracy of TAPIS's class on the given boxes {correct / len(g['annotations']):.4f}")
for key in res:
    pj = OUT / f"{key}.json"; json.dump(res[key], open(pj, "w"))
    out = ev.main_per_task(gtp, str(pj), "instruments", "mIoU", masks_path=None); m = out[0] if isinstance(out, tuple) else out
    json.dump(m, open(OUT / f"{key}_metrics.json", "w"), indent=1)
    print(key, {k: round(100 * v, 2) for k, v in m.items() if k in ("mIoU", "IoU", "mcIoU")}); print("   per class IoU:", [round(100 * m[c + "-IoU"], 1) for c in CL])
