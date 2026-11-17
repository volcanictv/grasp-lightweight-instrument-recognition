"""EndoVis 2018 (ISINet split): ISINet (our Mask R-CNN rerun, no temporal module) and MATIS (released EndoVis model) with the ANNOTATED masks. Every annotated instrument takes the class
distribution of the prediction that overlaps its mask most (mask IoU) and is painted with its annotated mask; scored with TAPIS's own evaluator, like the TAPIS rows. The raw predictions are scored too as a check.
usage (tapis_venv): python endovis_given_masks_match.py {isinet|matis}   (inputs in ~/ev_given and ~/ev_tapis_gtmask/val_anns_rle.json; outputs in ~/ev_given/out_<model>)"""
import json, os, sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mc
sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox")); sys.path.insert(0, os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox/tapis"))
import tapis.evaluate.main_eval as ev

model = sys.argv[1]; H = Path(os.path.expanduser("~/ev_given")); OUT = H / ("out_" + model); OUT.mkdir(parents=True, exist_ok=True)
gtp = os.path.expanduser("~/ev_tapis_gtmask/val_anns_rle.json"); g = json.load(open(gtp))
fix = lambda s: {"size": list(s["size"]), "counts": s["counts"].decode() if isinstance(s["counts"], bytes) else s["counts"]}
dec = lambda r: mc.decode({"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}).astype(bool)

if model == "isinet":
    preds = json.load(open(H / "isinet_preds.json"))
else:
    import torch
    sys.path.insert(0, str(H))
    from matis_compute_all_iou import getrealbox
    mp = json.load(open(H / "matis_preds_tools.json")); seg = {f["file_name"]: f["segments"] for f in torch.load(H / "matis_features_val.pth", map_location="cpu")["features"]}
    preds = {}
    for file, v in mp.items():
        jpg = file.rsplit(".", 1)[0] + ".jpg"; ins = []
        for box in v["bboxes"]:
            if box["bbox"] == [0.0, 0.0, 0.0, 0.0]: continue
            rle = fix(seg[file][getrealbox(seg[file].keys(), "{} {} {} {}".format(*box["bbox"]))]); x, y, w, h = mc.toBbox({"size": rle["size"], "counts": rle["counts"].encode()}).tolist()
            ins.append({"bbox": [x, y, x + w, y + h], "instruments_score_dist": list(box["prob_tools"])[:7], "segment": rle})
        preds[jpg] = {"instances": ins}
for im in g["images"]: preds.setdefault(im["file_name"], {"instances": []})


def run(pred_dict, name):
    pj = OUT / (name + ".json"); json.dump(pred_dict, open(pj, "w"))
    out = ev.main_per_task(gtp, str(pj), "instruments", "mIoU", masks_path=None); m = out[0] if isinstance(out, tuple) else out
    json.dump(m, open(OUT / (name + "_metrics.json"), "w"), indent=1)
    print(model, name, {k: round(100 * v, 2) for k, v in m.items() if k in ("mIoU", "IoU", "mcIoU")}); print("   per class:", {k.replace("-IoU", ""): round(100 * v, 1) for k, v in m.items() if k.endswith("-IoU")})


run({k: preds[k] for k in preds if k in {i["file_name"] for i in g["images"]}}, "check_raw_predictions")
res = {}; cov50 = 0; none = 0; acc = []
for a in g["annotations"]:
    gr = fix(a["rle"] if isinstance(a.get("rle"), dict) else a["segmentation"]); gm = dec(gr); best, bi = None, -1.0
    for c in preds[a["image_name"]]["instances"]:
        pm = dec(fix(c["segment"])); inter = float((gm & pm).sum()); iou = inter / max(1.0, float(gm.sum() + pm.sum() - inter))
        if iou > bi: best, bi = c, iou
    cov50 += bi >= 0.5; none += (best is None or bi <= 0)
    dist = best["instruments_score_dist"] if best is not None and bi > 0 else [1 / 7.0] * 7
    acc.append(int(np.argmax(dist)) == int(a["category_id"]) - 1)
    x0, y0, w, h = [float(v) for v in a["bbox"]]
    res.setdefault(a["image_name"], {"instances": []})["instances"].append({"bbox": [x0, y0, x0 + w, y0 + h], "instruments_score_dist": dist, "segment": gr})
for im in g["images"]: res.setdefault(im["file_name"], {"instances": []})
n = len(g["annotations"])
print(model, "annotated instruments %d: best prediction overlaps at mask IoU>=0.5 for %d (%.1f%%), no overlap at all %d; class accuracy of the matched prediction %.4f" % (n, cov50, 100.0 * cov50 / n, none, np.mean(acc)))
run(res, "annotated_masks_matched")
