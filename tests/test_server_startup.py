from types import SimpleNamespace

import numpy as np
import pytest

from frigate_igpu_detector import server


def _raise(exc: Exception):
    def inner(*_args, **_kwargs):
        raise exc

    return inner


def test_missing_backends_are_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "BIND", "tcp://127.0.0.1:0")
    monkeypatch.setattr(server, "BACKEND", "ncnn-vulkan")
    monkeypatch.setattr(
        server.FrigateZMQServer, "_init_vulkan", _raise(RuntimeError("no vulkan"))
    )
    monkeypatch.setattr(
        server.FrigateZMQServer, "_get_core", _raise(server.BackendUnavailable("no openvino"))
    )

    with pytest.raises(server.BackendUnavailable) as excinfo:
        server.FrigateZMQServer()
    assert "no usable inference backend" in str(excinfo.value)


def test_vulkan_failure_falls_back_to_openvino(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "BIND", "tcp://127.0.0.1:0")
    monkeypatch.setattr(server, "BACKEND", "ncnn-vulkan")
    monkeypatch.setattr(
        server.FrigateZMQServer, "_init_vulkan", _raise(RuntimeError("no vulkan"))
    )
    monkeypatch.setattr(server.FrigateZMQServer, "_get_core", lambda self: object())

    instance = server.FrigateZMQServer()
    try:
        assert instance.backend == "openvino"
        assert instance.model is None
        assert "no vulkan" in instance.backend_fallback_reason
    finally:
        instance.close()


def test_invalid_status_bind_does_not_abort_server(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "BIND", "tcp://127.0.0.1:0")
    monkeypatch.setattr(server, "BACKEND", "openvino")
    monkeypatch.setattr(server, "STATUS_BIND", "not-a-valid-bind")
    monkeypatch.setattr(server.FrigateZMQServer, "_get_core", lambda self: object())

    instance = server.FrigateZMQServer()
    try:
        assert instance.backend == "openvino"
        assert instance.status_server is None
    finally:
        instance.close()


def test_model_fallback_updates_runtime_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "BIND", "tcp://127.0.0.1:0")
    monkeypatch.setattr(server, "BACKEND", "ncnn-vulkan")
    monkeypatch.setattr(
        server.FrigateZMQServer, "_init_vulkan", _raise(RuntimeError("no vulkan"))
    )
    monkeypatch.setattr(server.FrigateZMQServer, "_get_core", lambda self: object())

    instance = server.FrigateZMQServer()
    try:
        instance.model = SimpleNamespace(name="model.onnx", backend="openvino")
        instance._sync_model_state("model.onnx", available=True)
        snapshot = instance.state.snapshot()
        assert snapshot["backend"] == {
            "configured": "ncnn-vulkan",
            "active": "openvino",
            "fallback": True,
        }
    finally:
        instance.close()


def test_missing_model_keeps_old_model_state(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "BIND", "tcp://127.0.0.1:0")
    monkeypatch.setattr(server, "BACKEND", "openvino")
    monkeypatch.setattr(server.FrigateZMQServer, "_get_core", lambda self: object())

    instance = server.FrigateZMQServer()
    try:
        instance.model = SimpleNamespace(name="old.onnx", backend="openvino")
        instance.state.set_model("old.onnx", True, True)
        assert instance._load_cached_model("missing.onnx") is False
        assert instance.state.snapshot()["model"] == {
            "name": "old.onnx",
            "loaded": True,
            "available": False,
        }
    finally:
        instance.close()


class _FakeSocket:
    def __init__(self) -> None:
        self.sent = []

    def send_multipart(self, parts):
        self.sent.append(parts)

    def close(self, linger=0):
        pass


class _BadModel:
    name = "bad.onnx"
    backend = "openvino"

    def infer(self, tensor):
        return np.zeros((1, 1), dtype=np.float32)


def test_invalid_inference_result_is_not_counted_as_latency(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "BIND", "tcp://127.0.0.1:0")
    monkeypatch.setattr(server, "BACKEND", "openvino")
    monkeypatch.setattr(server.FrigateZMQServer, "_get_core", lambda self: object())

    instance = server.FrigateZMQServer()
    try:
        instance.model = _BadModel()
        instance.socket = _FakeSocket()
        header = {"model_type": "yologeneric", "shape": [1, 3, 320, 320], "dtype": "float32"}
        tensor = np.zeros((1, 3, 320, 320), dtype=np.float32)
        instance._handle_inference([b"{}", tensor.tobytes()], header)
        snapshot = instance.state.snapshot()
        assert snapshot["inference"]["requests_total"] == 1
        assert snapshot["inference"]["errors_total"] == 1
        assert snapshot["inference"]["rejected_total"] == 1
        assert snapshot["inference"]["window"] == 0
        assert snapshot["inference"]["last_ms"] is None
    finally:
        instance.close()


def test_backend_failure_returns_nonzero_from_main(monkeypatch):
    monkeypatch.setattr(
        server, "FrigateZMQServer", _raise(server.BackendUnavailable("no backend"))
    )
    assert server.main() == 2
