#!/usr/bin/python3
"""Frigate ZMQ remote detector server (multi-backend).

Wire protocol follows Frigate v0.18.x ``frigate/detectors/plugins/zmq_ipc.py``:

- model request:      ``[JSON]``
- model transfer:     ``[JSON, ONNX bytes]``
- inference:          ``[JSON, contiguous tensor bytes]``
- inference response: ``[JSON(shape=[20,6], dtype=float32), float32 bytes]``

Backends (``FRIGATE_ZMQ_BACKEND``):

- ``openvino``    OpenVINO CPU plugin (fallback, works on any x86 CPU)
- ``ncnn-vulkan`` ncnn + Vulkan on an AMD/Intel GPU (fp16), automatic
                  ONNX -> ncnn conversion through pnnx, cached on disk

YOLO post-processing is adapted from Frigate v0.18.x ``frigate/util/model.py``
(MIT License, Copyright (c) Frigate, Inc.).
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zmq

from . import __version__

LOG = logging.getLogger("frigate_igpu_detector")
MODEL_DIR = Path(os.environ.get("FRIGATE_ZMQ_MODEL_DIR", "/var/lib/frigate-detector/models"))
BIND = os.environ.get("FRIGATE_ZMQ_BIND", "tcp://0.0.0.0:5555")
BACKEND = os.environ.get("FRIGATE_ZMQ_BACKEND", "openvino").strip().lower()
CPU_THREADS = max(1, int(os.environ.get("FRIGATE_ZMQ_CPU_THREADS", "4")))
NCNN_THREADS = max(1, int(os.environ.get("FRIGATE_ZMQ_NCNN_THREADS", "1")))
GPU_INDEX = int(os.environ.get("FRIGATE_ZMQ_GPU_INDEX", "-1"))
NCNN_CACHE_DIR = MODEL_DIR / "ncnn"
MAX_MODEL_BYTES = max(1, int(os.environ.get("FRIGATE_ZMQ_MAX_MODEL_MB", "512"))) * 1024 * 1024
MAX_TENSOR_BYTES = 128 * 1024 * 1024
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.onnx$")
KNOWN_BACKENDS = ("openvino", "ncnn-vulkan")


# ---------------------------------------------------------------------------
# YOLO post-processing (same semantics as Frigate)
# ---------------------------------------------------------------------------


def _xyxy_to_xywh_for_nms(boxes: np.ndarray) -> np.ndarray:
    """Frigate's corner-to-top-left/size conversion."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xywh = boxes.copy()
    xywh[:, 2] -= xywh[:, 0]
    xywh[:, 3] -= xywh[:, 1]
    return xywh


def _post_process_multipart_yolo(
    output_list: list[np.ndarray], width: int, height: int
) -> np.ndarray:
    # Same anchor/stride decoder as Frigate v0.18.x.
    if len(output_list) != 3:
        raise ValueError(
            f"Frigate multipart YOLO decoding expects 3 output heads, got {len(output_list)}"
        )

    anchors = [
        [(12, 16), (19, 36), (40, 28)],
        [(36, 75), (76, 55), (72, 146)],
        [(142, 110), (192, 243), (459, 401)],
    ]
    stride_map = {0: 8, 1: 16, 2: 32}
    all_boxes: list[list[float]] = []
    all_scores: list[float] = []
    all_class_ids: list[int] = []

    for i, output in enumerate(output_list):
        if output.ndim != 4:
            raise ValueError(f"YOLO output head {i} must be rank-4, got {output.shape}")
        bs, _, ny, nx = output.shape
        if bs != 1:
            raise ValueError(f"YOLO output head {i} batch must be 1, got {bs}")
        stride = stride_map[i]
        anchor_set = anchors[i]
        num_anchors = len(anchor_set)
        output = output.reshape(bs, num_anchors, 85, ny, nx)
        output = output.transpose(0, 1, 3, 4, 2)[0]

        for a_idx, (anchor_w, anchor_h) in enumerate(anchor_set):
            for y in range(ny):
                for x in range(nx):
                    pred = output[a_idx, y, x]
                    class_probs = pred[5:]
                    class_id = int(np.argmax(class_probs))
                    class_conf = float(class_probs[class_id])
                    conf = class_conf * float(pred[4])
                    if conf < CONF_THRESHOLD:
                        continue
                    dx, dy, dw, dh = pred[:4]
                    bx = ((float(dx) * 2.0 - 0.5) + x) * stride
                    by = ((float(dy) * 2.0 - 0.5) + y) * stride
                    bw = ((float(dw) * 2.0) ** 2) * anchor_w
                    bh = ((float(dh) * 2.0) ** 2) * anchor_h
                    x1 = max(0.0, bx - bw / 2)
                    y1 = max(0.0, by - bh / 2)
                    x2 = min(float(width), bx + bw / 2)
                    y2 = min(float(height), by + bh / 2)
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(conf)
                    all_class_ids.append(class_id)

    detections = np.zeros((20, 6), dtype=np.float32)
    if not all_boxes:
        return detections

    indices = cv2.dnn.NMSBoxes(
        _xyxy_to_xywh_for_nms(np.asarray(all_boxes, dtype=np.float32)),
        all_scores,
        score_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )
    indices = np.asarray(indices).reshape(-1).astype(np.int64) if len(indices) else []
    for i, idx in enumerate(indices[:20]):
        x1, y1, x2, y2 = all_boxes[int(idx)]
        detections[i] = [
            all_class_ids[int(idx)],
            all_scores[int(idx)],
            y1 / height,
            x1 / width,
            y2 / height,
            x2 / width,
        ]
    return detections


def post_process_yolo(output_list: list[np.ndarray], width: int, height: int) -> np.ndarray:
    """Equivalent of Frigate v0.18.x ``post_process_yolo()``."""
    if len(output_list) > 1:
        return _post_process_multipart_yolo(output_list, width, height)

    predictions = np.squeeze(np.asarray(output_list[0]))
    if predictions.ndim != 2:
        raise ValueError(
            f"Expected single YOLO output to squeeze to rank 2; got {predictions.shape}"
        )

    # Frigate's NMS-head decoder: (candidate_count, class_scores) with xywh first.
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
    # This deployment targets COCO-80 YOLO models. Their Frigate yolo-generic
    # output has 4 box values + 80 class scores = 84 columns.
    if predictions.shape[1] != 84:
        raise ValueError(
            f"Expected COCO-80 yolo-generic output with 84 columns (4 box + 80 class), got {predictions.shape}"
        )

    scores_all = np.max(predictions[:, 4:], axis=1)
    keep = scores_all > CONF_THRESHOLD
    predictions = predictions[keep, :]
    scores = scores_all[keep]
    detections = np.zeros((20, 6), dtype=np.float32)
    if predictions.size == 0:
        return detections

    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = predictions[:, :4]
    boxes_xyxy = np.empty_like(boxes, dtype=np.float32)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    indices = cv2.dnn.NMSBoxes(
        _xyxy_to_xywh_for_nms(boxes_xyxy),
        scores,
        score_threshold=CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
    )
    indices = np.asarray(indices).reshape(-1).astype(np.int64) if len(indices) else []
    for i, idx in enumerate(indices[:20]):
        bbox = boxes_xyxy[int(idx)]
        detections[i] = [
            int(class_ids[int(idx)]),
            float(scores[int(idx)]),
            float(bbox[1]) / height,
            float(bbox[0]) / width,
            float(bbox[3]) / height,
            float(bbox[2]) / width,
        ]
    return detections


# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------


def _tensor_shape(value_info: Any) -> tuple[int | None, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return ()
    dims: list[int | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(None)
    return tuple(dims)


def read_onnx_io(path: Path) -> tuple[tuple[int, ...], list[tuple[int | None, ...]]]:
    """Read static input shape and declared output shapes from an ONNX file.

    Only the graph metadata is used, so this works without OpenVINO and without
    running shape inference. Dynamic output dimensions are returned as ``None``
    and validated on the first inference instead.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(inputs) != 1:
        raise ValueError(f"Expected one model input, found {len(inputs)}")

    input_shape = _tensor_shape(inputs[0])
    if len(input_shape) != 4 or any(dim is None for dim in input_shape):
        raise ValueError(f"Expected a static rank-4 input shape, got {input_shape}")
    if input_shape[1] not in (1, 3, 4):
        raise ValueError(
            f"Input shape {input_shape} does not look like NCHW; Frigate sends NCHW tensors"
        )

    outputs = list(model.graph.output)
    if len(outputs) not in (1, 3):
        raise ValueError(
            f"Expected one NMS-head output or three YOLO heads, found {len(outputs)}"
        )
    output_shapes = [_tensor_shape(item) for item in outputs]
    if len(output_shapes) == 1:
        shape = output_shapes[0]
        if shape and shape[0] is not None and shape[0] != 1:
            raise ValueError(f"Expected batch size 1 in the output shape, got {shape}")
        if shape and all(dim is not None for dim in shape) and 84 not in shape[1:]:
            raise ValueError(
                f"Expected a single COCO-80 raw YOLO output such as [1,84,N] or [1,N,84], got {shape}"
            )
    return input_shape, output_shapes


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_pnnx() -> str | None:
    """Locate the pnnx executable.

    The pip wheel's console script imports torch, which is only needed for the
    Python API. The real static binary ships inside the package and works on
    its own, so prefer the release binary / packaged ELF over the console script.
    """
    override = os.environ.get("FRIGATE_ZMQ_PNNX")
    if override:
        return override if os.access(override, os.X_OK) else None

    found = shutil.which("pnnx")
    if found:
        # A real binary (downloaded release) is preferred over the pip wrapper.
        try:
            with open(found, "rb") as handle:
                if handle.read(4) == b"\x7fELF":
                    return found
        except OSError:
            pass

    try:
        import importlib.util

        spec = importlib.util.find_spec("pnnx")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if spec and spec.origin:
            locations.append(str(Path(spec.origin).parent))
        for location in locations:
            candidate = Path(location) / "pnnx"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    except Exception:
        pass
    return found


def _cleanup_stale_conversions() -> None:
    if not NCNN_CACHE_DIR.is_dir():
        return
    now = time.time()
    for candidate in NCNN_CACHE_DIR.glob(".work-*"):
        try:
            if now - candidate.stat().st_mtime > 3600:
                shutil.rmtree(candidate, ignore_errors=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class OpenVINOModel:
    backend = "openvino"

    def __init__(self, name: str, path: Path, core: Any):
        self.name = name
        self.path = path
        self.sha256 = _sha256_file(path)
        model = core.read_model(str(path))
        if len(model.inputs) != 1:
            raise ValueError(f"Expected one model input, found {len(model.inputs)}")
        if len(model.outputs) not in (1, 3):
            raise ValueError(
                f"Expected one NMS-head output or three YOLO heads, found {len(model.outputs)}"
            )

        self.compiled = core.compile_model(
            model,
            "CPU",
            {
                # THROUGHPUT + a single stream keeps one request in flight
                # (the REP loop is serial) while allowing all requested threads.
                "PERFORMANCE_HINT": "THROUGHPUT",
                "NUM_STREAMS": 1,
                "INFERENCE_NUM_THREADS": CPU_THREADS,
            },
        )
        self.input_port = self.compiled.input(0)
        self.input_shape = tuple(int(x) for x in self.input_port.shape)
        if len(self.input_shape) != 4 or any(x <= 0 for x in self.input_shape):
            raise ValueError(f"Expected a static rank-4 input shape, got {self.input_shape}")
        if self.input_shape[1] not in (1, 3, 4):
            raise ValueError(
                f"Input shape {self.input_shape} does not look like NCHW; Frigate sends NCHW"
            )
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        self.output_ports = list(self.compiled.outputs)
        self.output_shapes = [tuple(int(x) for x in port.shape) for port in self.output_ports]
        if len(self.output_shapes) == 1:
            out_shape = self.output_shapes[0]
            if len(out_shape) != 3 or out_shape[0] != 1 or 84 not in out_shape[1:]:
                raise ValueError(
                    f"Expected a single COCO-80 raw YOLO output such as [1,84,N] or [1,N,84], got {out_shape}"
                )
        self.infer_request = self.compiled.create_infer_request()
        LOG.info(
            "Loaded ONNX model=%s sha256=%s input=%s outputs=%s backend=openvino threads=%d",
            self.name,
            self.sha256,
            self.input_shape,
            self.output_shapes,
            CPU_THREADS,
        )

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        if tuple(tensor.shape) != self.input_shape:
            raise ValueError(
                f"Frigate tensor shape {tuple(tensor.shape)} does not match ONNX input {self.input_shape}"
            )
        if tensor.dtype != np.float32:
            raise ValueError(f"Expected float32 input from Frigate, got {tensor.dtype}")
        result = self.infer_request.infer({self.input_port: tensor})
        outputs = [np.asarray(result[port]) for port in self.output_ports]
        return post_process_yolo(outputs, self.width, self.height)


class NcnnVulkanModel:
    """ncnn + Vulkan backend (fp16).

    The ONNX file Frigate ships is converted with pnnx once and cached under
    ``<model dir>/ncnn``; the cache key includes the ONNX sha256, so changing a
    model triggers a re-conversion automatically.
    """

    backend = "ncnn-vulkan"

    def __init__(self, name: str, path: Path, gpu_index: int):
        import ncnn

        self.ncnn = ncnn
        self.name = name
        self.path = path
        self.sha256 = _sha256_file(path)
        self.input_shape, output_shapes = read_onnx_io(path)
        if len(output_shapes) != 1:
            raise ValueError(
                "ncnn-vulkan backend only supports single-output YOLO models; "
                f"this model has {len(output_shapes)} outputs. Use FRIGATE_ZMQ_BACKEND=openvino."
            )
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        param_path, bin_path = self._ensure_ncnn_model()
        net = ncnn.Net()
        net.opt.use_vulkan_compute = True
        net.opt.use_fp16_packed = True
        net.opt.use_fp16_storage = True
        net.opt.use_fp16_arithmetic = True
        # The REP loop runs one request at a time. Extra ncnn CPU worker threads
        # only spin while the GPU works (measured: 6 threads -> 33 ms CPU per
        # inference vs 3 ms with 1 thread).
        net.opt.num_threads = NCNN_THREADS
        net.opt.lightmode = True
        net.set_vulkan_device(gpu_index)
        if net.load_param(str(param_path)) != 0:
            raise RuntimeError(f"ncnn failed to load param {param_path}")
        if net.load_model(str(bin_path)) != 0:
            raise RuntimeError(f"ncnn failed to load model {bin_path}")
        input_names = list(net.input_names())
        output_names = list(net.output_names())
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"ncnn model exposes inputs={input_names} outputs={output_names}; "
                "the sidecar requires exactly one of each"
            )
        self.net = net
        self.input_name = input_names[0]
        self.output_name = output_names[0]

        # Warm up now so the first real inference does not blow Frigate's
        # 1000 ms request timeout.
        started = time.monotonic()
        raw = self._infer_raw(np.zeros((3, self.height, self.width), dtype=np.float32))
        warmup = time.monotonic() - started
        if warmup > 5.0:
            raise RuntimeError(
                f"ncnn Vulkan warmup took {warmup:.1f}s; this looks like the llvmpipe "
                "software rasteriser rather than a real GPU"
            )
        if raw.ndim != 2 or 84 not in raw.shape:
            raise RuntimeError(f"Unexpected ncnn output shape {raw.shape}; expected one 84-column head")
        LOG.info(
            "Loaded ONNX model=%s sha256=%s input=%s outputs=%s backend=ncnn-vulkan "
            "gpu=%d threads=%d fp16=1 warmup=%.2fs",
            self.name,
            self.sha256,
            self.input_shape,
            list(output_shapes),
            gpu_index,
            NCNN_THREADS,
            warmup,
        )

    @property
    def cache_key(self) -> str:
        return f"{Path(self.name).stem}-{self.sha256[:12]}"

    def _cache_paths(self) -> tuple[Path, Path]:
        key = self.cache_key
        return (
            NCNN_CACHE_DIR / f"{key}.ncnn.param",
            NCNN_CACHE_DIR / f"{key}.ncnn.bin",
        )

    def _ensure_ncnn_model(self) -> tuple[Path, Path]:
        param_path, bin_path = self._cache_paths()
        if (
            param_path.is_file()
            and bin_path.is_file()
            and param_path.stat().st_size > 0
            and bin_path.stat().st_size > 0
        ):
            LOG.info("Using cached ncnn model %s", param_path.name)
            return param_path, bin_path

        pnnx = find_pnnx()
        if not pnnx:
            raise RuntimeError(
                "pnnx not found; install the pnnx release binary or set FRIGATE_ZMQ_PNNX"
            )
        NCNN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        work_dir = NCNN_CACHE_DIR / f".work-{os.getpid()}-{time.time_ns()}"
        work_dir.mkdir()
        try:
            source = work_dir / "model.onnx"
            shutil.copyfile(self.path, source)
            shape_arg = "inputshape=[{}]".format(",".join(str(x) for x in self.input_shape))
            started = time.monotonic()
            proc = subprocess.run(
                [pnnx, str(source), shape_arg],
                cwd=str(work_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
            output = proc.stdout.decode("utf-8", "replace") if proc.stdout else ""
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pnnx conversion failed (rc={proc.returncode}): {output[-800:]}"
                )
            converted_param = work_dir / "model.ncnn.param"
            converted_bin = work_dir / "model.ncnn.bin"
            if not converted_param.is_file() or not converted_bin.is_file():
                raise RuntimeError(f"pnnx produced no ncnn model: {output[-800:]}")
            os.replace(converted_param, param_path)
            os.replace(converted_bin, bin_path)
            LOG.info(
                "Converted ONNX to ncnn in %.2fs: %s",
                time.monotonic() - started,
                param_path.name,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        self._prune_old_cache()
        return param_path, bin_path

    def _prune_old_cache(self) -> None:
        """Drop ncnn files for older versions of the same model name."""
        stem = Path(self.name).stem
        keep_prefix = self.cache_key
        for candidate in NCNN_CACHE_DIR.glob(f"{stem}-*.ncnn.*"):
            if candidate.name.startswith(keep_prefix):
                continue
            try:
                candidate.unlink()
                LOG.info("Removed stale ncnn cache file %s", candidate.name)
            except OSError:
                LOG.warning("Could not remove stale ncnn cache file %s", candidate)

    def _infer_raw(self, chw: np.ndarray) -> np.ndarray:
        ncnn = self.ncnn
        # ncnn's Python binding maps a contiguous (channels, height, width)
        # array onto Mat(c, h, w); Frigate already gives us NCHW.
        mat = ncnn.Mat(np.ascontiguousarray(chw, dtype=np.float32))
        extractor = self.net.create_extractor()
        try:
            if extractor.input(self.input_name, mat) != 0:
                raise RuntimeError(f"ncnn rejected input {self.input_name}")
            ret, out = extractor.extract(self.output_name)
            if ret != 0:
                raise RuntimeError(f"ncnn extract({self.output_name}) failed with code {ret}")
            result = np.array(out, dtype=np.float32, copy=True)
        finally:
            # Release the extractor before the next request; ncnn's blob
            # allocator must not outlive it.
            del extractor
        return result

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        if tuple(tensor.shape) != self.input_shape:
            raise ValueError(
                f"Frigate tensor shape {tuple(tensor.shape)} does not match ONNX input {self.input_shape}"
            )
        if tensor.dtype != np.float32:
            raise ValueError(f"Expected float32 input from Frigate, got {tensor.dtype}")
        raw = self._infer_raw(tensor[0])
        if raw.ndim != 2 or 84 not in raw.shape:
            raise RuntimeError(f"Unexpected ncnn output shape {raw.shape}")
        return post_process_yolo([raw], self.width, self.height)


# ---------------------------------------------------------------------------
# ZMQ server
# ---------------------------------------------------------------------------


class FrigateZMQServer:
    def __init__(self) -> None:
        if BACKEND not in KNOWN_BACKENDS:
            raise RuntimeError(
                f"Unsupported FRIGATE_ZMQ_BACKEND={BACKEND!r}; expected one of {KNOWN_BACKENDS}"
            )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.core: Any = None
        self.backend = BACKEND
        self.gpu_ready = False
        self.gpu_index = -1
        if self.backend == "ncnn-vulkan":
            try:
                self._init_vulkan()
            except Exception:
                LOG.exception("Vulkan/GPU initialisation failed; falling back to OpenVINO CPU")
                self.backend = "openvino"
        if self.backend == "openvino":
            LOG.info("Inference backend: OpenVINO CPU, threads=%d", CPU_THREADS)
        self.model: OpenVINOModel | NcnnVulkanModel | None = None
        self._ncnn_failed: set[str] = set()
        self._last_reject = ""
        self._last_reject_at = 0.0
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 2)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        # Wake periodically so SIGTERM can stop the loop without waiting for a
        # request (systemd otherwise SIGKILLs us after TimeoutStopSec).
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.socket.bind(BIND)
        LOG.info("Frigate ZMQ server %s listening on %s", __version__, BIND)
        LOG.info("Model cache directory: %s", MODEL_DIR)
        self._autoload_cached_model()

    def _get_core(self) -> Any:
        if self.core is None:
            try:
                import openvino as ov
            except ImportError as exc:
                raise RuntimeError(
                    "The OpenVINO backend requires the optional dependency; install it with "
                    "`pip install 'frigate-igpu-detector[openvino]'`"
                ) from exc
            core = ov.Core()
            if "CPU" not in core.available_devices:
                raise RuntimeError(f"OpenVINO CPU plugin unavailable: {core.available_devices}")
            self.core = core
        return self.core

    def _init_vulkan(self) -> None:
        import ncnn

        if not Path("/dev/dri").exists():
            raise RuntimeError("/dev/dri not available inside this container")
        render_nodes = sorted(Path("/dev/dri").glob("renderD*"))
        accessible = [node for node in render_nodes if os.access(node, os.R_OK | os.W_OK)]
        if not accessible:
            raise RuntimeError(
                "no accessible /dev/dri/renderD* device; the service user must be in the "
                f"'render' group (found {[str(node) for node in render_nodes] or 'none'})"
            )
        _cleanup_stale_conversions()
        ncnn.create_gpu_instance()
        count = int(ncnn.get_gpu_count())
        if count < 1:
            raise RuntimeError("no Vulkan device found (check Mesa/RADV drivers)")
        index = GPU_INDEX
        if index < 0:
            index = int(ncnn.get_default_gpu_index())
        if index < 0 or index >= count:
            index = 0
        self.gpu_index = index
        self.gpu_ready = True
        LOG.info(
            "Inference backend: ncnn Vulkan, devices=%d, gpu=%d, threads=%d, cache=%s",
            count,
            index,
            NCNN_THREADS,
            NCNN_CACHE_DIR,
        )

    def _build_model(self, name: str, path: Path) -> OpenVINOModel | NcnnVulkanModel:
        if self.backend == "ncnn-vulkan":
            try:
                sha = _sha256_file(path)
            except Exception:
                sha = None
            if sha is not None and sha in self._ncnn_failed:
                LOG.warning(
                    "ncnn-vulkan previously failed for this model (sha=%s); using OpenVINO CPU",
                    sha[:12],
                )
                return OpenVINOModel(name, path, self._get_core())
            try:
                return NcnnVulkanModel(name, path, self.gpu_index)
            except Exception:
                if sha is not None:
                    self._ncnn_failed.add(sha)
                LOG.exception(
                    "ncnn-vulkan model load failed for %s; using OpenVINO CPU", name
                )
        return OpenVINOModel(name, path, self._get_core())

    @staticmethod
    def _safe_model_name(value: Any) -> str:
        name = str(value or "")
        if not MODEL_NAME_RE.fullmatch(name) or Path(name).name != name:
            raise ValueError(f"Unsafe or unsupported model filename: {name!r}")
        return name

    def _load_cached_model(self, name: str) -> bool:
        path = MODEL_DIR / name
        if not path.is_file():
            return False
        # Only quarantine the file when it is structurally not a readable ONNX.
        # Backend failures (missing OpenVINO, missing pnnx, GPU unavailable) are
        # environmental and must not move a valid model aside.
        try:
            read_onnx_io(path)
        except Exception:
            LOG.exception("Cached model is not a readable ONNX: %s", path)
            self._quarantine(path)
            return False
        try:
            self.model = self._build_model(name, path)
            return True
        except Exception:
            LOG.exception(
                "Failed to load cached model %s (backend error); leaving the file in place", path
            )
            self.model = None
            return False

    @staticmethod
    def _quarantine(path: Path) -> None:
        quarantine = path.with_name(f"{path.name}.invalid-{int(time.time())}")
        try:
            path.replace(quarantine)
            LOG.warning("Moved invalid cached model to %s", quarantine)
        except OSError:
            LOG.exception("Could not quarantine invalid model file %s", path)

    def _autoload_cached_model(self) -> None:
        """Load the newest cached model at startup.

        Frigate's ZMQ detector only asks for a model during its own startup, so
        without this a sidecar restart leaves Frigate sending inference requests
        that cannot be served until Frigate itself restarts.
        """
        candidates = sorted(
            (path for path in MODEL_DIR.glob("*.onnx") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            if self._load_cached_model(path.name):
                LOG.info("Auto-loaded cached model at startup: %s", path.name)
                return
        LOG.info("No cached model in %s; waiting for Frigate model transfer", MODEL_DIR)

    def _send_json(self, data: dict[str, Any]) -> None:
        self.socket.send(json.dumps(data, separators=(",", ":")).encode("utf-8"))

    def _handle_model_request(self, header: dict[str, Any]) -> None:
        name = self._safe_model_name(header.get("model_name"))
        if self.model is None or self.model.name != name:
            loaded = self._load_cached_model(name)
        else:
            loaded = True
        # Frigate v0.18.x requires BOTH flags true before it considers the model ready.
        self._send_json({"model_available": bool(loaded), "model_loaded": bool(loaded)})
        if not loaded:
            LOG.info("Model %s not cached; waiting for Frigate model transfer", name)

    def _handle_model_data(self, frames: list[bytes], header: dict[str, Any]) -> None:
        name = self._safe_model_name(header.get("model_name"))
        if len(frames) != 2:
            self._send_json(
                {"model_saved": False, "model_loaded": False, "error": "expected two frames"}
            )
            return
        model_data = frames[1]
        if not model_data or len(model_data) > MAX_MODEL_BYTES:
            self._send_json(
                {"model_saved": False, "model_loaded": False, "error": "model size rejected"}
            )
            return

        target = MODEL_DIR / name
        temp = MODEL_DIR / f".{name}.{os.getpid()}.uploading"
        try:
            with temp.open("wb") as handle:
                handle.write(model_data)
                handle.flush()
                os.fsync(handle.fileno())
            # Compile before replacing the last known-good cached model.
            candidate = self._build_model(name, temp)
            os.replace(temp, target)
            candidate.path = target
            self.model = candidate
            LOG.info(
                "Model transfer complete: %s bytes=%d sha256=%s",
                name,
                len(model_data),
                candidate.sha256,
            )
            self._send_json({"model_saved": True, "model_loaded": True})
        except Exception as exc:
            LOG.exception("Model transfer/load failed for %s", name)
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                LOG.exception("Failed to remove temporary model %s", temp)
            self._send_json(
                {"model_saved": False, "model_loaded": False, "error": str(exc)[:300]}
            )

    def _send_zero_result(self, message: str) -> None:
        now = time.monotonic()
        if message != self._last_reject or now - self._last_reject_at > 5.0:
            LOG.error("Inference request rejected: %s", message)
            self._last_reject = message
            self._last_reject_at = now
        zero = np.zeros((20, 6), dtype=np.float32)
        meta = {"shape": [20, 6], "dtype": "float32"}
        self.socket.send_multipart(
            [json.dumps(meta, separators=(",", ":")).encode("utf-8"), zero.tobytes(order="C")]
        )

    def _handle_inference(self, frames: list[bytes], header: dict[str, Any]) -> None:
        if len(frames) != 2:
            self._send_zero_result(f"expected 2 frames, received {len(frames)}")
            return
        if self.model is None:
            self._send_zero_result("no model is loaded")
            return
        try:
            if header.get("model_type") not in (
                "yologeneric",
                "yolo-generic",
                "ModelTypeEnum.yologeneric",
            ):
                raise ValueError(
                    f"Only yolo-generic is supported, got {header.get('model_type')!r}"
                )
            shape_raw = header.get("shape")
            if not isinstance(shape_raw, list) or not shape_raw or any(
                not isinstance(x, int) or x <= 0 for x in shape_raw
            ):
                raise ValueError(f"Invalid tensor shape in header: {shape_raw!r}")
            shape = tuple(shape_raw)
            dtype = np.dtype(header.get("dtype", "float32"))
            if dtype != np.dtype(np.float32):
                raise ValueError(f"Expected float32 tensor from Frigate, got {dtype}")
            expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if expected_bytes <= 0 or expected_bytes > MAX_TENSOR_BYTES or len(frames[1]) != expected_bytes:
                raise ValueError(
                    f"Tensor byte count mismatch: shape={shape} requires {expected_bytes}, "
                    f"received {len(frames[1])}"
                )
            tensor = np.frombuffer(frames[1], dtype=dtype).reshape(shape)
            result = self.model.infer(tensor)
            if result.shape != (20, 6) or result.dtype != np.float32:
                raise ValueError(f"Postprocessor returned invalid result {result.shape}/{result.dtype}")
            meta = {"shape": [20, 6], "dtype": "float32"}
            self.socket.send_multipart(
                [json.dumps(meta, separators=(",", ":")).encode("utf-8"), result.tobytes(order="C")]
            )
        except Exception as exc:
            LOG.exception("Inference/postprocess failed (backend=%s)", self.backend)
            self._send_zero_result(str(exc))

    def run(self) -> None:
        while not STOP_REQUESTED:
            try:
                frames = self.socket.recv_multipart()
                if not frames:
                    self._send_json({"error": "empty request"})
                    continue
                try:
                    header = json.loads(frames[0].decode("utf-8"))
                except Exception as exc:
                    self._send_json({"error": f"invalid JSON header: {exc}"})
                    continue
                if not isinstance(header, dict):
                    self._send_json({"error": "header must be a JSON object"})
                elif header.get("model_request") is True:
                    self._handle_model_request(header)
                elif header.get("model_data") is True:
                    self._handle_model_data(frames, header)
                else:
                    self._handle_inference(frames, header)
            except zmq.Again:
                # recv timeout: loop around and re-check STOP_REQUESTED
                continue
            except zmq.ContextTerminated:
                break
            except zmq.ZMQError:
                if STOP_REQUESTED:
                    break
                LOG.exception("ZMQ server error")
                raise
            except Exception:
                LOG.exception("Unexpected server-loop error")
                # A REP socket must always send a reply after receiving a request.
                try:
                    self._send_json({"error": "internal server error"})
                except Exception:
                    LOG.exception("Could not send error reply")

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            # ncnn requires every Net to be destroyed before the Vulkan instance.
            self.model = None
            gc.collect()
            if self.gpu_ready:
                try:
                    import ncnn

                    ncnn.destroy_gpu_instance()
                except Exception:
                    LOG.exception("Could not destroy the Vulkan instance")
            LOG.info("Server stopped")


STOP_REQUESTED = False


def _request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOG.info("Received signal %s; shutting down", signum)


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("FRIGATE_ZMQ_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    LOG.info("frigate-igpu-detector %s starting", __version__)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    server: FrigateZMQServer | None = None
    try:
        server = FrigateZMQServer()
        server.run()
        return 0
    except Exception:
        LOG.exception("Fatal server error")
        return 1
    finally:
        if server is not None:
            server.close()


if __name__ == "__main__":
    raise SystemExit(main())
