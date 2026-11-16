"""Prepares TAPIS's released two-fold evaluation on our 1 Hz-style frames: (1) the annotated fold json renamed to the 30 fps frame names of the released predictions (keyframes paired in time order per case),
(2) a symlink farm so every 30 fps frame name of the fold's frame list points to the nearest available image (piecewise map through the keyframe pairs)."""
import json, os, sys
import numpy as np
from pathlib import Path
H = Path(os.path.expanduser("~/tapis_folds/data/GraSP")); R = Path(os.path.expanduser("~/Desktop/Classification Surgurical Tools/GraSP")); FR = R / "frames-001" / "frames"
for f in (1, 2):
    p = json.load(open(H / "annotations" / f"fold{f}_val_preds.json")); g = json.load(open(R / "annotations" / f"grasp_short-term_fold{f}.json"))
    th, ou = {}, {}
    for i in p["images"]: c, fn = i["file_name"].split("/"); th.setdefault(c, []).append(int(fn[:-4]))
    for i in g["images"]: c, fn = i["file_name"].split("/"); ou.setdefault(c, []).append(int(fn[:-4]))
    name_map, pairs = {}, {}
    for c in th:
        n = sorted(th[c]); k = sorted(ou[c]); assert len(n) == len(k), (c, len(n), len(k)); pairs[c] = (np.array(n, float), np.array(k, float))
        for kk, nn in zip(k, n): name_map[f"{c}/{kk:05d}.jpg"] = (f"{c}/{nn:09d}.jpg", nn)
    for im in g["images"]: im["file_name"], im["frame_num"] = name_map[im["file_name"]]
    for a in g["annotations"]: a["image_name"] = name_map[a["image_name"]][0]
    json.dump(g, open(H / "annotations" / f"grasp_short-term_fold{f}.json", "w"))
    made = 0; missing = 0
    for line in open(H / "frame_lists" / f"fold{f}.csv"):
        parts = line.split()
        if len(parts) < 4 or parts[0] == "original_vido_id": continue
        c = parts[0]
        if c not in pairs: continue
        path = parts[3]; n = int(path.split("/")[1][:-4]); ns, ks = pairs[c]
        if n <= ns[0]: kf = ks[0] + (n - ns[0]) / 45.0
        elif n >= ns[-1]: kf = ks[-1] + (n - ns[-1]) / 45.0
        else: kf = float(np.interp(n, ns, ks))
        k = max(0, int(round(kf))); src = FR / c / f"{k:05d}.jpg"
        if not src.exists():
            k = min(k, len(os.listdir(FR / c)) - 1); src = FR / c / f"{k:05d}.jpg"
            if not src.exists(): missing += 1; continue
        dst = H / "frames" / path; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() and not dst.is_symlink(): os.symlink(src, dst); made += 1
    print(f"fold{f}: images {len(g['images'])} annotations {len(g['annotations'])}; links created {made}, frames without a source {missing}", flush=True)
