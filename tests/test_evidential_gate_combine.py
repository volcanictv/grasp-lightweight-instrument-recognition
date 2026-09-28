import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evidential_gate_fold1_calibrate import evidential_track_predict


def test_unanimous_confident_frames_pick_that_class():
    # 3 frames, 4 members, 3 classes, every member confidently votes class 1 on every frame
    det = np.full((3, 4, 3), -5.0)
    det[:, :, 1] = 5.0
    assert evidential_track_predict(det) == 1


def test_one_confident_frame_outweighs_several_unsure_frames():
    # 1 very confident frame for class 0, 4 barely-leaning frames for class 1
    confident = np.full((1, 4, 3), -5.0)
    confident[:, :, 0] = 8.0
    unsure = np.full((4, 4, 3), 0.0)
    unsure[:, :, 1] = 0.3
    det = np.concatenate([confident, unsure], axis=0)
    assert evidential_track_predict(det) == 0
