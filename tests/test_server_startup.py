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


def test_backend_failure_returns_nonzero_from_main(monkeypatch):
    monkeypatch.setattr(
        server, "FrigateZMQServer", _raise(server.BackendUnavailable("no backend"))
    )
    assert server.main() == 2
