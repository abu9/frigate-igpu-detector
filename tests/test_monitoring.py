import json
import urllib.error
import urllib.request

import pytest

from frigate_igpu_detector.monitoring import (
    RuntimeState,
    StatusHTTPServer,
    parse_status_bind,
)


def test_parse_status_bind():
    assert parse_status_bind("") is None
    assert parse_status_bind("127.0.0.1:9180") == ("127.0.0.1", 9180)
    assert parse_status_bind("tcp://0.0.0.0:9180") == ("0.0.0.0", 9180)
    with pytest.raises(ValueError):
        parse_status_bind("127.0.0.1")
    with pytest.raises(ValueError):
        parse_status_bind("127.0.0.1:not-a-port")


def test_snapshot_state():
    state = RuntimeState("ncnn-vulkan", "ncnn-vulkan", False)
    state.set_model("model.onnx", True, True)
    state.record_request()
    state.record_request()
    state.record_rejected()
    state.record_error()
    state.record_inference(40.0)
    state.record_inference(30.0)
    state.record_inference(10.0)

    snapshot = state.snapshot()
    assert snapshot["backend"] == {
        "configured": "ncnn-vulkan",
        "active": "ncnn-vulkan",
        "fallback": False,
    }
    assert snapshot["model"] == {
        "name": "model.onnx",
        "loaded": True,
        "available": True,
    }
    assert snapshot["inference"]["requests_total"] == 2
    assert snapshot["inference"]["rejected_total"] == 1
    assert snapshot["inference"]["errors_total"] == 1
    assert snapshot["inference"]["last_ms"] == 10.0
    assert snapshot["inference"]["p50_ms"] == 30.0
    assert snapshot["inference"]["window"] == 3
    assert "cpu_percent" in snapshot["live"]
    assert "gpu_busy_percent" in snapshot["live"]


def test_status_http_endpoint():
    state = RuntimeState("ncnn-vulkan", "ncnn-vulkan", False)
    state.set_model("model.onnx", False, True)
    server = StatusHTTPServer(("127.0.0.1", 0), state)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(base + "/status", timeout=5) as response:
            assert response.status == 200
            payload = json.loads(response.read())
        assert payload["model"]["loaded"] is False
        assert "inference" in payload

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(base + "/status?require_model=1", timeout=5)
        assert excinfo.value.code == 503

        state.set_model("model.onnx", True, True)
        with urllib.request.urlopen(base + "/status?require_model=1", timeout=5) as response:
            assert response.status == 200

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(base + "/nope", timeout=5)
        assert excinfo.value.code == 404

        request = urllib.request.Request(base + "/status", method="POST", data=b"")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(request, timeout=5)
        assert excinfo.value.code == 405
    finally:
        server.stop()
