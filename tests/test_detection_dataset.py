import json

from PIL import Image

from surgical_ai.data.detection_dataset import GraspDetectionDataset


def _make_dataset(tmp_path):
    doc = {
        "categories": [{"id": 1, "name": "ToolA"}, {"id": 2, "name": "ToolB"}],
        "images": [{"id": 1, "file_name": "CASE001/1.jpg", "width": 20, "height": 20, "video_name": "CASE001"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [2, 3, 4, 5]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]},
        ],
    }
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "grasp_short-term_train.json").write_text(json.dumps(doc))

    frames_dir = tmp_path / "frames-001" / "frames" / "CASE001"
    frames_dir.mkdir(parents=True)
    Image.new("RGB", (20, 20), color=(0, 0, 0)).save(frames_dir / "1.jpg")

    return GraspDetectionDataset(tmp_path, "train", transform=None)


def test_boxes_converted_to_xyxy_and_labels_offset_for_background(tmp_path):
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    image, target = ds[0]

    boxes = target["boxes"].tolist()
    labels = target["labels"].tolist()

    # category_id=2 -> index 1 -> label 2 (0 reserved for background); bbox
    # [2,3,4,5] (x,y,w,h) -> [2,3,6,8] (x1,y1,x2,y2).
    assert [2.0, 3.0, 6.0, 8.0] in boxes
    assert 2 in labels
    # category_id=1 -> index 0 -> label 1; bbox [0,0,1,1] -> [0,0,1,1].
    assert [0.0, 0.0, 1.0, 1.0] in boxes
    assert 1 in labels
    assert 0 not in labels  # background label never appears in ground truth


def test_class_names_ordered_matches_category_ids(tmp_path):
    ds = _make_dataset(tmp_path)
    assert ds.class_names_ordered() == ["ToolA", "ToolB"]
