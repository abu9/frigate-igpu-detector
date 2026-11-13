"""Local ZMQ protocol and inference probe for the Frigate detector."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zmq

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        parts = tuple(int(x) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x1,y1,x2,y2")
    return parts  # type: ignore[return-value]


def labels_for(score_row: np.ndarray) -> str:
    class_id = int(score_row[0])
    return COCO80[class_id] if 0 <= class_id < len(COCO80) else f"class_{class_id}"


def decode_reply(parts: list[bytes]) -> np.ndarray:
    if len(parts) == 1:
        if len(parts[0]) == 20 * 6 * 4:
            return np.frombuffer(parts[0], dtype=np.float32).reshape((20, 6))
        raise RuntimeError(f"single-frame reply has {len(parts[0])} bytes")
    meta = json.loads(parts[0])
    return np.frombuffer(parts[1], dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])


def _send_recv(sock: zmq.Socket, parts: list[bytes], timeout_ms: int) -> list[bytes]:
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.send_multipart(parts)
    return sock.recv_multipart()


def model_request(sock: zmq.Socket, name: str, timeout_ms: int) -> dict[str, Any]:
    parts = _send_recv(
        sock,
        [json.dumps({"model_request": True, "model_name": name}).encode()],
        timeout_ms,
    )
    if len(parts) != 1:
        raise RuntimeError(f"unexpected model_request reply: {len(parts)} frames")
    return json.loads(parts[0])


def model_transfer(
    sock: zmq.Socket, name: str, path: Path, timeout_ms: int
) -> dict[str, Any]:
    data = path.read_bytes()
    parts = _send_recv(
        sock,
        [
            json.dumps({"model_data": True, "model_name": name}).encode(),
            data,
        ],
        timeout_ms,
    )
    if len(parts) != 1:
        raise RuntimeError(f"unexpected model_data reply: {len(parts)} frames")
    return json.loads(parts[0])


def infer(
    sock: zmq.Socket, tensor: np.ndarray, timeout_ms: int
) -> tuple[np.ndarray, float]:
    header = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype.name),
        "model_type": "yologeneric",
    }
    start = time.perf_counter()
    parts = _send_recv(
        sock,
        [json.dumps(header).encode(), tensor.tobytes(order="C")],
        timeout_ms,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return decode_reply(parts), elapsed_ms


def read_model_shape(model_path: Path) -> tuple[int, ...]:
    from .server import read_onnx_io

    shape, _ = read_onnx_io(model_path)
    return tuple(int(x) for x in shape)


def preprocess(
    image_path: Path,
    shape: tuple[int, ...],
    crop: tuple[int, int, int, int] | None,
) -> np.ndarray:
    _, _, height, width = shape
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if crop is not None:
        x1, y1, x2, y2 = crop
        image = image[y1:y2, x1:x2]
    else:
        ih, iw = image.shape[:2]
        side = min(ih, iw)
        y0 = (ih - side) // 2
        x0 = (iw - side) // 2
        image = image[y0 : y0 + side, x0 : x0 + side]
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    tensor = np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)
    if tuple(tensor.shape) != tuple(shape):
        raise RuntimeError(f"preprocessed shape {tensor.shape} != model input {shape}")
    return tensor


def run_probe(
    *,
    endpoint: str,
    model_path: Path,
    model_name: str | None = None,
    image_path: Path | None = None,
    crop: tuple[int, int, int, int] | None = None,
    runs: int = 5,
    timeout_ms: int = 10000,
    model_timeout_ms: int = 180000,
) -> dict[str, Any]:
    """Run a probe and return a JSON-serialisable result.

    This function never raises for a remote/protocol failure; the caller gets
    ``ok=False`` and ``error`` instead. Local programming errors still raise.
    """
    if not model_path.is_file():
        return {"ok": False, "error": f"model not found: {model_path}"}
    name = model_name or model_path.name

    result: dict[str, Any] = {
        "ok": False,
        "endpoint": endpoint,
        "model": name,
        "image": str(image_path) if image_path else None,
        "crop": list(crop) if crop else None,
        "runs": 0,
        "before": None,
        "after": None,
        "transfer": None,
        "input_shape": None,
        "latency_ms": None,
        "nonzero_detections_last": None,
        "detections": [],
        "error": None,
    }

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(endpoint)
        result["before"] = model_request(sock, name, model_timeout_ms)
        state = result["before"]
        if not (state.get("model_available") and state.get("model_loaded")):
            if not state.get("model_available"):
                started = time.perf_counter()
                result["transfer"] = model_transfer(sock, name, model_path, model_timeout_ms)
                result["transfer"]["elapsed_ms"] = round(
                    (time.perf_counter() - started) * 1000.0, 1
                )
                if not (
                    result["transfer"].get("model_saved")
                    and result["transfer"].get("model_loaded")
                ):
                    result["error"] = "model transfer failed"
                    return result
            else:
                result["error"] = "server reports model available but not loaded"
                return result

        result["after"] = model_request(sock, name, model_timeout_ms)
        state = result["after"]
        if not (state.get("model_available") and state.get("model_loaded")):
            result["error"] = "model is not ready after transfer"
            return result

        shape = read_model_shape(model_path)
        result["input_shape"] = list(shape)

        if image_path is None:
            result["ok"] = True
            return result

        tensor = preprocess(image_path, shape, crop)

        for _ in range(2):
            infer(sock, tensor, timeout_ms)

        times: list[float] = []
        nonzero: list[int] = []
        dets = np.zeros((20, 6), dtype=np.float32)
        for _ in range(max(1, runs)):
            dets, elapsed_ms = infer(sock, tensor, timeout_ms)
            times.append(elapsed_ms)
            nonzero.append(int(np.count_nonzero(dets[:, 1] > 0)))

        if dets.shape != (20, 6) or dets.dtype != np.float32:
            result["error"] = f"bad inference result: {dets.shape}/{dets.dtype}"
            return result

        detections: list[dict[str, Any]] = []
        for row in dets:
            class_id, score, y1, x1, y2, x2 = [float(x) for x in row]
            if score <= 0:
                continue
            detections.append(
                {
                    "class_id": int(class_id),
                    "label": labels_for(row),
                    "score": round(score, 3),
                    "box": [
                        round(x1, 3),
                        round(y1, 3),
                        round(x2, 3),
                        round(y2, 3),
                    ],
                }
            )

        times_sorted = sorted(times)
        p50 = statistics.median(times_sorted)
        p95 = times_sorted[max(0, int(round(0.95 * (len(times_sorted) - 1))))]
        result.update(
            {
                "ok": True,
                "runs": len(times),
                "latency_ms": {
                    "min": round(min(times), 2),
                    "p50": round(p50, 2),
                    "p95": round(p95, 2),
                    "max": round(max(times), 2),
                },
                "nonzero_detections_last": nonzero[-1],
                "detections": detections,
            }
        )
        return result
    except zmq.Again as exc:
        result["error"] = f"ZMQ timeout: {exc}"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        sock.close(0)
        ctx.term()


def print_result(result: dict[str, Any]) -> None:
    if result.get("before") is not None:
        print(f"before: {json.dumps(result['before'])}")
    if result.get("transfer") is not None:
        print(f"transfer: {json.dumps(result['transfer'])}")
    if result.get("after") is not None:
        print(f"after:  {json.dumps(result['after'])}")
    if result.get("input_shape"):
        print(f"model input shape: {tuple(result['input_shape'])}")
    if result.get("detections"):
        print("detections (last run):")
        for det in result["detections"]:
            box = ",".join(f"{x:.3f}" for x in det["box"])
            print(f"  {det['label']:16s} score={det['score']:.3f} box=({box})")
    elif result.get("runs"):
        print("detections (last run):\n  (none)")
    if result.get("latency_ms"):
        print(f"SUMMARY {json.dumps(result, separators=(',', ':'))}")
    if not result.get("ok"):
        print(f"ERROR: {result.get('error')}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fzd probe")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--crop", default=None)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=10000)
    parser.add_argument("--model-timeout", type=int, default=180000)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | argparse.Namespace | None = None) -> int:
    if isinstance(argv, argparse.Namespace):
        args = argv
    else:
        args = _parser().parse_args(argv)

    model_path = Path(args.model)
    result = run_probe(
        endpoint=args.endpoint,
        model_path=model_path,
        model_name=args.model_name,
        image_path=Path(args.image) if args.image else None,
        crop=parse_crop(args.crop),
        runs=args.runs,
        timeout_ms=args.timeout,
        model_timeout_ms=args.model_timeout,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_result(result)
    if result.get("ok"):
        return 0
    return 2 if "ZMQ timeout" in str(result.get("error")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
