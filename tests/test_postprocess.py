import numpy as np
import pytest

from frigate_igpu_detector.server import post_process_yolo


def _output(boxes, scores, classes, anchors=2100):
    """Build a (1, 84, anchors) raw YOLO output."""
    raw = np.zeros((1, 84, anchors), dtype=np.float32)
    for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        raw[0, :4, index] = box
        raw[0, 4 + class_id, index] = score
    return raw


def test_empty_output_returns_zero_padding():
    raw = np.zeros((1, 84, 2100), dtype=np.float32)
    result = post_process_yolo([raw], 320, 320)
    assert result.shape == (20, 6)
    assert result.dtype == np.float32
    assert not result.any()


def test_single_person_box_is_normalized():
    # cx, cy, w, h in model pixels; the box becomes 128..192 in a 320 input.
    raw = _output([[160.0, 160.0, 64.0, 64.0]], [0.9], [0])
    result = post_process_yolo([raw], 320, 320)

    class_id, score, y1, x1, y2, x2 = result[0]
    assert class_id == 0
    assert score == pytest.approx(0.9, abs=1e-5)
    assert (y1, x1, y2, x2) == pytest.approx((0.4, 0.4, 0.6, 0.6), abs=1e-5)
    assert not result[1:].any()


def test_nms_keeps_the_highest_score():
    raw = _output(
        [
            [160.0, 160.0, 64.0, 64.0],
            [161.0, 161.0, 64.0, 64.0],
        ],
        [0.9, 0.8],
        [0, 0],
    )
    result = post_process_yolo([raw], 320, 320)
    assert (result[:, 1] > 0).sum() == 1
    assert result[0, 1] == pytest.approx(0.9, abs=1e-5)


def test_below_threshold_is_dropped():
    raw = _output([[160.0, 160.0, 64.0, 64.0]], [0.39], [0])
    result = post_process_yolo([raw], 320, 320)
    assert not result.any()


def test_wrong_class_count_is_rejected():
    raw = np.zeros((1, 6, 4), dtype=np.float32)  # NMS-style output
    with pytest.raises(ValueError, match="84 columns"):
        post_process_yolo([raw], 320, 320)
