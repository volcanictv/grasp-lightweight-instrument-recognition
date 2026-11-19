"""Edits to the TAPIS checkout's EndoVis 2017 dataset class (tapis/datasets/endovis_2017.py), the same fixes the 2018 class needed (docs/reports/endovis_gtbox/tapis_protocol.md):
predicted-box features keyed by the box tuple, three return values of the image preprocessing of the base class, a frame mapping that tolerates frames without annotations, and an integer video
number in the frame name. Keeps a copy of the original next to the file and is safe to run twice.

    python scripts/patch_tapis_endovis2017.py ~/baselines/GraSP/TAPIS/tapis/datasets/endovis_2017.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

EDITS = [
    ("""    def keyframe_mapping(self, video_idx, sec_idx, sec):
        return sec
""",
     """    def keyframe_mapping(self, video_idx, sec_idx, sec):
        # the frame lists hold annotated frames only, so frame numbers have gaps and the list index differs from the frame number
        if not hasattr(self, "_sec_to_idx"):
            self._sec_to_idx = [{int(p.rsplit("_frame", 1)[1].split(".")[0]): j for j, p in enumerate(paths)} for paths in self._image_paths]
        return self._sec_to_idx[video_idx][sec]
"""),
    ("""    def frame_num_joining(self, video_num, sec):
        return self.frame_name_joining(video_num, sec)
""",
     """    def frame_num_joining(self, video_num, sec):
        return f"seq{video_num}_frame{sec:0{self.zero_fill}d}.{self.image_type}"
"""),
    ("""                        rpn_box_key = " ".join(map(str,box_labels['bbox']))
                        if rpn_box_key not in box_features.keys():
                            rpn_box_key = utils.get_best_features(box_labels["bbox"],box_features)
                        try:
                            features = np.array(box_features[rpn_box_key])
                            rpn_features.append(features)
                        except:

                            rpn_features.append(np.zeros(self.cfg.FEATURES.DIM_FEATURES))
""",
     """                        rpn_box_key = tuple(box_labels['bbox'])
                        try:
                            rpn_features.append(np.array(box_features[rpn_box_key]))
                        except KeyError:
                            if self.cfg.ENDOVIS_DATASET.INCLUDE_GT and box_labels['is_gt']:
                                rpn_box_key = utils.get_best_features(box_labels["bbox"], box_features)
                                rpn_features.append(np.array(box_features[rpn_box_key]))
                            else:
                                raise ValueError(f"Predicted box {box_labels['bbox']} missing in features of {complete_name}")
"""),
    ("""        imgs, boxes = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )
""",
     """        imgs, boxes, _ = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes, image=None
        )
"""),
]


def main() -> None:
    path = Path(sys.argv[1])
    text = path.read_text()
    if "_sec_to_idx" in text:
        print("already patched")
        return
    shutil.copy(path, path.with_suffix(".py.orig"))
    for old, new in EDITS:
        assert text.count(old) == 1, f"edit not found exactly once: {old[:60]!r}"
        text = text.replace(old, new)
    path.write_text(text)
    print(f"patched {path} ({len(EDITS)} edits); original saved as {path.with_suffix('.py.orig')}")


if __name__ == "__main__":
    main()
