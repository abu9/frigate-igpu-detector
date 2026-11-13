"""Read-only HTTP status endpoint for the detector."""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import __version__

LOG = logging.getLogger("frigate_igpu_detector.monitoring")
LATENCY_WINDOW = 300


def parse_status_bind(value: str) -> tuple[str, int] | None:
    """Parse ``host:port`` or ``tcp://host:port``; empty disables the endpoint."""
    value = (value or "").strip()
    if not value:
        return None
    if value.startswith("tcp://"):
        value = value[len("tcp://") :]
    host, _, port = value.rpartition(":")
    if not host or not port:
        raise ValueError(f"invalid FRIGATE_ZMQ_STATUS_BIND={value!r}; expected host:port")
    try:
        return host, int(port)
    except ValueError as exc:
        raise ValueError(f"invalid port in FRIGATE_ZMQ_STATUS_BIND={value!r}") from exc


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _resident_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_memory_bytes() -> int | None:
    """Return this process' cgroup memory, not the whole container's."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                # cgroup v2
                value = _read_int(Path("/sys/fs/cgroup", path.lstrip("/"), "memory.current"))
                if value is not None:
                    return value
            elif "memory" in controllers.split(","):
                # cgroup v1
                value = _read_int(
                    Path("/sys/fs/cgroup/memory", path.lstrip("/"), "memory.usage_in_bytes")
                )
                if value is not None:
                    return value
    except (OSError, ValueError):
        pass
    return None


def _gpu_busy_percent() -> int | None:
    for path in sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")):
        value = _read_int(path)
        if value is not None:
            return value
    return None


class RuntimeState:
    """Thread-safe snapshot of backend/model/live/inference state."""

    def __init__(self, configured_backend: str, active_backend: str, fallback: bool) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._configured_backend = configured_backend
        self._active_backend = active_backend
        self._fallback = fallback
        self._model_name: str | None = None
        self._model_loaded = False
        self._model_available = False
        self._cpu_cores: float | None = None
        self._cpu_percent: float | None = None
        self._requests_total = 0
        self._errors_total = 0
        self._rejected_total = 0
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._last_latency: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._sample_cpu, name="fzd-cpu-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _sample_cpu(self) -> None:
        last_cpu = time.process_time()
        last_wall = time.monotonic()
        while not self._stop.wait(1.0):
            now_cpu = time.process_time()
            now_wall = time.monotonic()
            delta_wall = now_wall - last_wall
            if delta_wall > 0:
                cores = (now_cpu - last_cpu) / delta_wall
                with self._lock:
                    self._cpu_cores = round(cores, 3)
                    self._cpu_percent = round(cores * 100.0, 1)
            last_cpu = now_cpu
            last_wall = now_wall

    def set_backend(self, configured: str, active: str, fallback: bool) -> None:
        with self._lock:
            self._configured_backend = configured
            self._active_backend = active
            self._fallback = fallback

    def set_model(self, name: str | None, loaded: bool, available: bool) -> None:
        with self._lock:
            self._model_name = name
            self._model_loaded = loaded
            self._model_available = available

    def record_request(self) -> None:
        with self._lock:
            self._requests_total += 1

    def record_rejected(self) -> None:
        with self._lock:
            self._rejected_total += 1

    def record_error(self) -> None:
        with self._lock:
            self._errors_total += 1

    def record_inference(self, elapsed_ms: float) -> None:
        with self._lock:
            self._last_latency = float(elapsed_ms)
            self._latencies.append(float(elapsed_ms))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
            if latencies:
                p50 = statistics.median(latencies)
                p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
                last = self._last_latency
            else:
                p50 = p95 = last = None
            state = {
                "version": __version__,
                "backend": {
                    "configured": self._configured_backend,
                    "active": self._active_backend,
                    "fallback": self._fallback,
                },
                "model": {
                    "name": self._model_name,
                    "loaded": self._model_loaded,
                    "available": self._model_available,
                },
                "live": {
                    "cpu_percent": self._cpu_percent,
                    "cpu_cores": self._cpu_cores,
                    "gpu_busy_percent": None,
                    "cgroup_memory_bytes": None,
                    "resident_memory_bytes": None,
                },
                "inference": {
                    "requests_total": self._requests_total,
                    "errors_total": self._errors_total,
                    "rejected_total": self._rejected_total,
                    "last_ms": round(last, 2) if last is not None else None,
                    "p50_ms": round(p50, 2) if p50 is not None else None,
                    "p95_ms": round(p95, 2) if p95 is not None else None,
                    "window": len(latencies),
                },
            }
        state["live"]["gpu_busy_percent"] = _gpu_busy_percent()
        state["live"]["cgroup_memory_bytes"] = _cgroup_memory_bytes()
        state["live"]["resident_memory_bytes"] = _resident_memory_bytes()
        return state


class _StatusHandler(BaseHTTPRequestHandler):
    server_version = f"frigate-igpu-detector/{__version__}"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        if parsed.path != "/status":
            self._send_json(404, {"error": "not found"})
            return
        snapshot = self.server.state.snapshot()  # type: ignore[attr-defined]
        require_model = parse_qs(parsed.query).get("require_model", ["0"])[0].lower()
        status = 503 if require_model in ("1", "true", "yes") and not snapshot["model"]["loaded"] else 200
        self._send_json(status, snapshot)

    def do_HEAD(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "method not allowed"})

    def do_POST(self) -> None:  # noqa: N802
        self._send_json(405, {"error": "method not allowed"})

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.debug("status http: " + fmt, *args)


class StatusHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, bind: tuple[str, int], state: RuntimeState) -> None:
        self.state = state
        super().__init__(bind, _StatusHandler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="fzd-status-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
