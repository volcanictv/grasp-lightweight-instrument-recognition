"""Files TAPIS's EndoVis dataset class needs after scripts/endovis_tapis_prep.py (2018) or scripts/endovis2017_tapis_prep.py (2017) and match_annots_n_preds.py have run, in <work>/tapis_data:
frame lists (4 columns: video, video number, frame number, file), image_name on every annotation (the meters key on it), and an RLE-mask copy of the validation ground truth (TAPIS's
semantic-segmentation scorer decodes RLE, the Mask2Former JSONs hold polygons). The frame lists hold the frames that have at least one annotation.

Usage (titanxp, tapis_venv):
    python scripts/tapis_endovis2017_post.py --work ~/endovis_tapis_isinet                      # EndoVis 2018 (seq_N names, .jpg)
    python scripts/tapis_endovis2017_post.py --work ~/endovis_tapis2017_fold0 --dataset 2017     # EndoVis 2017 (seqN names, .png)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--dataset", choices=["2018", "2017"], default="2018")
    args = ap.parse_args()
    prefix, ext = (4, "jpg") if args.dataset == "2018" else (3, "png")
    data = args.work / "tapis_data"
    (data / "frame_lists").mkdir(exist_ok=True)
    for split in ("train", "val"):
        for kind in ("anns", "preds"):
            path = data / "annotations" / f"{split}_{kind}.json"
            d = json.loads(path.read_text())
            names = {im["id"]: im["file_name"] for im in d["images"]}
            for a in d["annotations"]:
                a["image_name"] = names[a["image_id"]]
            path.write_text(json.dumps(d))
            if kind == "anns":
                annotated = {a["image_id"] for a in d["annotations"]}
                frames = sorted({(im["video_name"], im["frame_num"]) for im in d["images"] if im["id"] in annotated}, key=lambda t: (int(t[0][prefix:]), t[1]))
                (data / "frame_lists" / f"{split}.csv").write_text("".join(f"{v} {int(v[prefix:])} {n} {v}_frame{n:03d}.{ext}\n" for v, n in frames))
                print(f"{split}: {len(frames)} frames", flush=True)
                if split == "val":
                    for a in d["annotations"]:
                        a["segmentation"] = a["rle"]
                    (data / "annotations" / "val_anns_rle.json").write_text(json.dumps(d))


if __name__ == "__main__":
    main()
