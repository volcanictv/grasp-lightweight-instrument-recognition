import numpy as np
import torch

from surgical_ai.data.segmentation_targets import draw_gaussian_peak, gaussian_radius, render_instance_targets


def test_gaussian_radius_positive_and_finite():
    r = gaussian_radius(height=20, width=30)
    assert r > 0
    assert np.isfinite(r)


def test_draw_gaussian_peak_centered_at_one():
    channel = np.zeros((20, 20), dtype=np.float32)
    draw_gaussian_peak(channel, center=(10, 10), radius=3)
    assert channel[10, 10] == 1.0
    assert channel.max() == 1.0


def test_draw_gaussian_peak_takes_max_not_overwrite():
    channel = np.zeros((20, 20), dtype=np.float32)
    draw_gaussian_peak(channel, center=(10, 10), radius=5)
    before = channel[10, 10]
    draw_gaussian_peak(channel, center=(11, 11), radius=1)  # smaller, nearby peak
    assert channel[10, 10] == before  # not decreased/overwritten by the second, weaker peak


def test_render_instance_targets_shapes():
    height, width, stride, num_classes = 64, 64, 4, 3
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    result = render_instance_targets([mask], [1], num_classes, height, width, stride)

    out_h, out_w = height // stride, width // stride
    assert result["semantic"].shape == (out_h, out_w)
    assert result["heatmap"].shape == (num_classes, out_h, out_w)
    assert result["offset"].shape == (2, out_h, out_w)
    assert result["offset_mask"].shape == (out_h, out_w)


def test_render_instance_targets_semantic_and_offset_mask_agree():
    height, width, stride, num_classes = 64, 64, 4, 3
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    result = render_instance_targets([mask], [2], num_classes, height, width, stride)

    fg = result["semantic"] > 0
    assert torch.equal(fg, result["offset_mask"])
    assert (result["semantic"][fg] == 3).all()  # label_idx 2 -> semantic id 3 (0=background)


def test_render_instance_targets_no_instances_is_all_background():
    height, width, stride, num_classes = 32, 32, 4, 3
    result = render_instance_targets([], [], num_classes, height, width, stride)
    assert result["semantic"].sum() == 0
    assert result["heatmap"].sum() == 0
    assert not result["offset_mask"].any()


def test_render_instance_targets_heatmap_peak_at_centroid():
    height, width, stride, num_classes = 64, 64, 4, 2
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # centroid at (30, 30) in full-res -> (7.5, 7.5) at stride 4
    result = render_instance_targets([mask], [0], num_classes, height, width, stride)
    peak_y, peak_x = np.unravel_index(result["heatmap"][0].argmax(), result["heatmap"][0].shape)
    assert abs(peak_x - 7.5) <= 1
    assert abs(peak_y - 7.5) <= 1
