from pathlib import Path

from frigate_igpu_detector.status import (
    _client_endpoint,
    _collect_models,
    _fmt_bytes,
    _parse_log_state,
)


def test_parse_log_state_prefers_latest_model():
    lines = [
        "Inference backend: ncnn Vulkan, devices=2, gpu=0",
        "Auto-loaded cached model at startup: old.onnx",
        "Inference backend: OpenVINO CPU, threads=6",
        "Model transfer complete: new.onnx",
    ]
    state = _parse_log_state(lines)
    assert state["backend"] == "OpenVINO CPU"
    assert state["loaded_model"] == "new.onnx"
    assert state["fallback"] is False


def test_parse_log_state_detects_fallback():
    lines = [
        "Vulkan/GPU initialisation failed; falling back to OpenVINO CPU: RuntimeError: no gpu",
        "Inference backend: OpenVINO CPU, threads=6",
    ]
    state = _parse_log_state(lines)
    assert state["fallback"] is True
    assert state["backend"] == "OpenVINO CPU"


def test_fmt_bytes():
    assert _fmt_bytes(None) == "n/a"
    assert _fmt_bytes(0) == "0 B"
    assert _fmt_bytes(1536) == "1.5 KiB"
    assert _fmt_bytes(1024 * 1024) == "1.0 MiB"


def test_client_endpoint_maps_wildcards():
    assert _client_endpoint("tcp://0.0.0.0:5555") == "tcp://127.0.0.1:5555"
    assert _client_endpoint("tcp://127.0.0.1:5555") == "tcp://127.0.0.1:5555"
    assert _client_endpoint(None) is None


def test_collect_models_hashes_and_sorts(tmp_path: Path):
    older = tmp_path / "old.onnx"
    newer = tmp_path / "new.onnx"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    # Make the order deterministic even on coarse filesystems.
    import os

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    models = _collect_models(tmp_path)
    assert [model["name"] for model in models] == ["new.onnx", "old.onnx"]
    assert all(len(model["sha256_prefix"]) == 12 for model in models)
