"""Score a SlowFast/TAPIS test run with the annotated masks (the 'given masks' protocol of the segmentation table).

Each annotated instrument takes the class distribution of the prediction it is matched to (by mask IoU for runs on the model's own proposals, by box IoU for given-box runs) and is painted with its
annotated mask; the result goes through TAPIS's own evaluator, so numbers are comparable with the TAPIS rows. Instruments without any overlapping prediction get a uniform distribution.
usage: score_given_masks.py --gt GT.json --preds epoch_0_preds_instruments.json --out DIR --match mask|box --tapis-dir CODE_DIR
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--preds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--match", choices=["mask", "box"], required=True); ap.add_argument("--tapis-dir", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.tapis_dir); sys.path.insert(0, os.path.join(a.tapis_dir, "tapis"))
    import tapis.evaluate.main_eval as ev
    from pycocotools import mask as mc

    fix = lambda s: {"size": list(s["size"]), "counts": s["counts"].decode() if isinstance(s["counts"], bytes) else s["counts"]}
    dec = lambda r: mc.decode({"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}).astype(bool)

    def box_iou(p, q):
        ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
        return i / max(1e-9, (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i)

    g = json.load(open(a.gt)); preds = json.load(open(a.preds)); res = {}; covered = 0; hits = []
    for an in g["annotations"]:
        gr = fix(an["rle"] if isinstance(an.get("rle"), dict) else an["segmentation"]); x0, y0, w, h = [float(v) for v in an["bbox"]]; box = [x0, y0, x0 + w, y0 + h]
        gm = dec(gr) if a.match == "mask" else None; best, bi = None, 0.0
        for c in preds.get(an["image_name"], {"instances": []})["instances"]:
            if a.match == "mask":
                pm = dec(fix(c["segment"])); inter = float((gm & pm).sum()); s = inter / max(1.0, float(gm.sum() + pm.sum() - inter))
            else:
                s = box_iou(box, c["bbox"])
            if s > bi: best, bi = c, s
        covered += best is not None
        dist = best["instruments_score_dist"] if best is not None else [1 / 7.0] * 7
        hits.append(int(np.argmax(dist)) == int(an["category_id"]) - 1)
        res.setdefault(an["image_name"], {"instances": []})["instances"].append({"bbox": box, "instruments_score_dist": dist, "segment": gr})
    for im in g["images"]: res.setdefault(im["file_name"], {"instances": []})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); pj = out / "annotated_masks_matched.json"; json.dump(res, open(pj, "w"))
    n = len(g["annotations"]); r = ev.main_per_task(a.gt, str(pj), "instruments", "mIoU", masks_path=None); m = r[0] if isinstance(r, tuple) else r
    summary = {"instruments": n, "matched_to_a_prediction": covered, "class_accuracy_of_matched_prediction": float(np.mean(hits)),
               "mIoU": m["mIoU"], "IoU": m["IoU"], "mcIoU": m["mcIoU"], "per_class_IoU": {k[:-4]: v for k, v in m.items() if k.endswith("-IoU") and k != "IoU"}, "match": a.match}
    json.dump({"summary": summary, "full": m}, open(out / "metrics.json", "w"), indent=1)
    print(f"{a.match}-matched: instruments {n}, with a prediction {covered}, class accuracy {np.mean(hits):.4f}")
    print({k: round(100 * summary[k], 2) for k in ("mIoU", "IoU", "mcIoU")}); print("per class IoU:", {k: round(100 * v, 1) for k, v in summary["per_class_IoU"].items()})


if __name__ == "__main__":
    main()
