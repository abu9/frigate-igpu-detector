import onnx
import pytest
from onnx import TensorProto, helper

from frigate_igpu_detector.server import read_onnx_io


def _save(tmp_path, input_dims, output_dims, name="model.onnx"):
    inputs = [helper.make_tensor_value_info("images", TensorProto.FLOAT, input_dims)]
    outputs = [helper.make_tensor_value_info("output0", TensorProto.FLOAT, output_dims)]
    node = helper.make_node("Identity", ["images"], ["output0"])
    graph = helper.make_graph([node], "g", inputs, outputs)
    model = helper.make_model(graph)
    path = tmp_path / name
    onnx.save(model, path)
    return path


def test_static_shapes(tmp_path):
    path = _save(tmp_path, [1, 3, 320, 320], [1, 84, 2100])
    input_shape, output_shapes = read_onnx_io(path)
    assert input_shape == (1, 3, 320, 320)
    assert output_shapes == [(1, 84, 2100)]


def test_dynamic_output_is_allowed(tmp_path):
    path = _save(tmp_path, [1, 3, 320, 320], [1, "anchors", 84])
    input_shape, output_shapes = read_onnx_io(path)
    assert input_shape == (1, 3, 320, 320)
    assert output_shapes == [(1, None, 84)]


def test_non_nchw_input_is_rejected(tmp_path):
    path = _save(tmp_path, [1, 320, 320, 3], [1, 84, 2100])
    with pytest.raises(ValueError, match="NCHW"):
        read_onnx_io(path)


def test_missing_84_columns_is_rejected(tmp_path):
    path = _save(tmp_path, [1, 3, 320, 320], [1, 6, 2100])
    with pytest.raises(ValueError, match="COCO-80"):
        read_onnx_io(path)
