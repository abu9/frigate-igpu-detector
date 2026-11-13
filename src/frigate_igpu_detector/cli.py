"""Command line entry point for frigate-igpu-detector."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import platform
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import __version__


def _ok(label: str, detail: str = "") -> None:
    print(f"[ OK ] {label}{': ' + detail if detail else ''}")


def _warn(label: str, detail: str = "") -> None:
    print(f"[WARN] {label}{': ' + detail if detail else ''}")


def _fail(label: str, detail: str = "") -> None:
    print(f"[FAIL] {label}{': ' + detail if detail else ''}")


def _parse_bind(value: str) -> tuple[str, int] | None:
    if not value.startswith("tcp://"):
        return None
    host, _, port = value[len("tcp://") :].rpartition(":")
    try:
        return host, int(port)
    except ValueError:
        return None


_UNIT_NAME = os.environ.get("FRIGATE_ZMQ_UNIT", "frigate-igpu-detector.service")


def _parse_env_blob(blob: str) -> dict[str, str]:
    """Parse the space-separated KEY=VALUE output of ``systemctl show -p Environment``."""
    env: dict[str, str] = {}
    for token in shlex.split(blob):
        if "=" in token:
            key, value = token.split("=", 1)
            env[key] = value
    return env


def _parse_proc_environ(raw: bytes) -> dict[str, str]:
    """Parse a NUL-separated ``/proc/<pid>/environ`` blob."""
    env: dict[str, str] = {}
    for item in raw.decode("utf-8", "replace").split("\0"):
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    return env


def _deployed_environment(unit: str) -> tuple[dict[str, str], str]:
    """Return the environment of the running unit, falling back to its unit file.

    The running process is preferred because it includes EnvironmentFile values
    and anything that was overridden at start time. The second return value
    describes where the values came from.
    """
    try:
        proc = subprocess.run(
            ["systemctl", "show", "-p", "MainPID", "--value", unit],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        pid = proc.stdout.strip()
        environ = Path(f"/proc/{pid}/environ") if pid and pid != "0" else None
        if environ is not None and environ.is_file():
            return _parse_proc_environ(environ.read_bytes()), f"running unit {unit}"
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["systemctl", "show", "-p", "Environment", "--value", unit],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        env = _parse_env_blob(proc.stdout.strip())
        if env:
            return env, f"unit {unit}"
    except Exception:
        pass

    return {}, "environment"


def _doctor() -> int:
    from . import server

    print(f"frigate-igpu-detector {__version__}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    print()

    failures = 0

    # Prefer the deployed unit's environment over the invoking shell. Otherwise
    # `fzd doctor` would report the built-in defaults (openvino, /var/lib/...)
    # even though the service is running with different values.
    deployed, config_source = _deployed_environment(_UNIT_NAME)
    cfg = {**os.environ, **deployed}
    backend = cfg.get("FRIGATE_ZMQ_BACKEND", server.BACKEND)
    bind_value = cfg.get("FRIGATE_ZMQ_BIND", server.BIND)
    model_dir = Path(cfg.get("FRIGATE_ZMQ_MODEL_DIR", str(server.MODEL_DIR)))
    cache_dir = model_dir / "ncnn"
    ncnn_threads = cfg.get("FRIGATE_ZMQ_NCNN_THREADS", str(server.NCNN_THREADS))
    cpu_threads = cfg.get("FRIGATE_ZMQ_CPU_THREADS", str(server.CPU_THREADS))

    # Device nodes and permissions
    dridir = Path("/dev/dri")
    nodes = sorted(dridir.glob("renderD*")) if dridir.is_dir() else []
    if not nodes:
        _fail("/dev/dri/renderD* not found", "pass the render node into the container")
        failures += 1
    else:
        accessible = [node for node in nodes if os.access(node, os.R_OK | os.W_OK)]
        if accessible:
            _ok("render node", ", ".join(str(node) for node in accessible))
        else:
            _fail(
                "render node exists but is not accessible",
                "add the service user to the 'render' group and map the GID",
            )
            failures += 1

    try:
        import grp

        group_names = {grp.getgrgid(gid).gr_name for gid in os.getgroups()}
        if "render" in group_names:
            _ok("group membership", "render")
        else:
            _warn(
                "current process is not in the 'render' group",
                "only a problem if the render node is not world-accessible",
            )
    except Exception as exc:
        _warn("could not read group membership", str(exc))

    # Vulkan
    if not shutil.which("vulkaninfo"):
        _warn("vulkaninfo not installed", "apt install vulkan-tools")
    else:
        try:
            proc = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            text = proc.stdout + proc.stderr
            devices = [
                line.split("=", 1)[1].strip()
                for line in text.splitlines()
                if line.strip().startswith("deviceName")
            ]
            hardware = [name for name in devices if "llvmpipe" not in name.lower()]
            if hardware:
                _ok("Vulkan devices", "; ".join(hardware))
            elif devices:
                _fail(
                    "only llvmpipe (software Vulkan) is visible",
                    "the GPU is not accessible from this container",
                )
                failures += 1
            else:
                _fail("no Vulkan device reported by vulkaninfo")
                failures += 1
        except Exception as exc:
            _fail("vulkaninfo failed", str(exc))
            failures += 1

    # ncnn + GPU
    try:
        import ncnn

        ncnn.create_gpu_instance()
        try:
            count = int(ncnn.get_gpu_count())
            if count < 1:
                _fail("ncnn sees no Vulkan device")
                failures += 1
            else:
                _ok("ncnn Vulkan", f"{count} device(s), default gpu {ncnn.get_default_gpu_index()}")
        finally:
            ncnn.destroy_gpu_instance()
    except Exception as exc:
        _fail("ncnn import/init failed", f"{type(exc).__name__}: {exc}")
        failures += 1

    # pnnx
    pnnx_override = cfg.get("FRIGATE_ZMQ_PNNX")
    if pnnx_override:
        pnnx = pnnx_override if os.access(pnnx_override, os.X_OK) else None
    else:
        pnnx = server.find_pnnx()
    if pnnx:
        _ok("pnnx", pnnx)
    else:
        _fail("pnnx not found", "download the release binary or set FRIGATE_ZMQ_PNNX")
        failures += 1

    print()

    # Configuration
    bind = _parse_bind(bind_value)
    print(f"config:   {config_source}")
    print(f"backend:  {backend}")
    print(f"bind:     {bind_value}")
    print(f"models:   {model_dir}")
    print(f"threads:  ncnn={ncnn_threads} cpu={cpu_threads}")
    if bind:
        probe_host = bind[0] if bind[0] not in ("0.0.0.0", "::", "*", "") else "127.0.0.1"
        try:
            with socket.create_connection((probe_host, bind[1]), timeout=1):
                _ok("server listening", f"{probe_host}:{bind[1]}")
        except OSError:
            _warn("nothing listening", f"{probe_host}:{bind[1]} (is the service running?)")

    # Model cache
    if model_dir.is_dir():
        models = sorted(model_dir.glob("*.onnx"))
        if models:
            _ok("cached models", ", ".join(path.name for path in models))
        else:
            _warn("no cached ONNX model", "Frigate sends it on first startup")
        writable = os.access(model_dir, os.W_OK)
        (_ok if writable else _warn)("model directory writable", str(model_dir))
        if cache_dir.is_dir():
            cached = list(cache_dir.glob("*.ncnn.param"))
            _ok("ncnn cache", f"{len(cached)} model(s)")
    else:
        _warn("model directory does not exist yet", str(model_dir))

    print()
    if failures:
        print(f"{failures} check(s) failed")
        return 1
    print("all checks passed")
    return 0


def _preflight() -> int:
    """Validate dependencies and configuration without starting the server.

    This is intentionally lightweight: it imports the optional backends but does
    not create a Vulkan instance. It is used as ``ExecStartPre`` so a missing
    backend fails the unit before the ZMQ port is bound.
    """
    from . import server

    deployed, config_source = _deployed_environment(_UNIT_NAME)
    cfg = {**os.environ, **deployed}
    backend = cfg.get("FRIGATE_ZMQ_BACKEND", server.BACKEND).strip().lower()
    bind_value = cfg.get("FRIGATE_ZMQ_BIND", server.BIND)
    model_dir = Path(cfg.get("FRIGATE_ZMQ_MODEL_DIR", str(server.MODEL_DIR)))

    failures = 0
    warnings = 0

    if backend not in server.KNOWN_BACKENDS:
        _fail("unsupported backend", f"{backend!r}; expected one of {server.KNOWN_BACKENDS}")
        failures += 1
    else:
        _ok("backend", backend)

    ncnn_ok = False
    try:
        import ncnn  # noqa: F401

        ncnn_ok = True
        _ok("ncnn")
    except Exception as exc:
        _fail("ncnn", f"{type(exc).__name__}: {exc}")

    openvino_ok = False
    try:
        import openvino as ov

        core = ov.Core()
        if "CPU" not in core.available_devices:
            raise RuntimeError(f"CPU plugin unavailable: {core.available_devices}")
        openvino_ok = True
        _ok("openvino", "CPU plugin")
    except Exception as exc:
        _fail("openvino", f"{type(exc).__name__}: {exc}")

    if backend == "openvino":
        if not openvino_ok:
            failures += 1
    elif backend == "ncnn-vulkan":
        if ncnn_ok:
            pass
        elif openvino_ok:
            _warn("ncnn missing; the server will use the OpenVINO CPU fallback")
            warnings += 1
        else:
            failures += 1
    else:
        # Unknown backend already counted above.
        pass

    bind = _parse_bind(bind_value)
    if bind is None:
        _fail("invalid FRIGATE_ZMQ_BIND", bind_value)
        failures += 1
    else:
        _ok("bind", bind_value)

    if model_dir.is_dir():
        if os.access(model_dir, os.W_OK):
            _ok("model directory writable", str(model_dir))
        else:
            _warn("model directory is not writable", str(model_dir))
            warnings += 1
    else:
        parent = model_dir.parent
        if parent.is_dir() and os.access(parent, os.W_OK):
            _warn("model directory does not exist yet", str(model_dir))
            warnings += 1
        else:
            _fail("model directory cannot be created", str(model_dir))
            failures += 1

    if not ncnn_ok or not openvino_ok:
        print(
            "hint: if ncnn/OpenVINO are installed system-wide, confirm "
            "'include-system-site-packages = true' in the venv's pyvenv.cfg"
        )

    print(f"config: {config_source}")
    if failures:
        print(f"preflight failed: {failures} required check(s) failed")
        return 1
    print("preflight passed" + (f" ({warnings} warning(s))" if warnings else ""))
    return 0


def _bench(args: argparse.Namespace) -> int:
    import numpy as np

    from . import server

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 2
    if args.cache_dir:
        server.NCNN_CACHE_DIR = Path(args.cache_dir)
        server.NCNN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    backend = args.backend or server.BACKEND
    gpu_ready = False
    model = None
    started = time.monotonic()
    try:
        if backend == "ncnn-vulkan":
            import ncnn

            ncnn.create_gpu_instance()
            gpu_ready = True
            index = server.GPU_INDEX
            if index < 0:
                index = int(ncnn.get_default_gpu_index())
            model = server.NcnnVulkanModel(model_path.name, model_path, index)
        elif backend == "openvino":
            import openvino as ov

            model = server.OpenVINOModel(model_path.name, model_path, ov.Core())
        else:
            print(f"unknown backend: {backend}", file=sys.stderr)
            return 2
        load_s = time.monotonic() - started

        input_shape = list(model.input_shape)
        rng = np.random.default_rng(0)
        tensor = rng.random(input_shape, dtype=np.float32)

        walls: list[float] = []
        cpus: list[float] = []
        for _ in range(max(1, args.runs)):
            wall0 = time.perf_counter()
            cpu0 = time.process_time()
            result = model.infer(tensor)
            cpus.append((time.process_time() - cpu0) * 1000)
            walls.append((time.perf_counter() - wall0) * 1000)
        if result.shape != (20, 6):
            raise RuntimeError(f"unexpected result shape {result.shape}")

        walls.sort()
        cpus.sort()
        summary = {
            "version": __version__,
            "backend": backend,
            "model": model_path.name,
            "input_shape": input_shape,
            "load_seconds": round(load_s, 2),
            "runs": len(walls),
            "latency_ms": {
                "min": round(walls[0], 2),
                "p50": round(statistics.median(walls), 2),
                "p95": round(walls[max(0, int(round(0.95 * (len(walls) - 1))))], 2),
                "max": round(walls[-1], 2),
            },
            "cpu_ms_per_inference": round(statistics.median(cpus), 2),
        }
        print(json.dumps(summary, indent=2))
    except Exception as exc:
        print(f"bench failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        # ncnn requires every Net to be destroyed before the Vulkan instance.
        model = None
        gc.collect()
        if gpu_ready:
            import ncnn

            ncnn.destroy_gpu_instance()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("FRIGATE_ZMQ_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="fzd",
        description="Frigate ZMQ remote detector server (OpenVINO CPU / ncnn Vulkan).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="run the ZMQ detector server (default)")
    subparsers.add_parser("doctor", help="check GPU, drivers, pnnx and model cache")
    subparsers.add_parser(
        "preflight", help="validate runtime dependencies before starting the server"
    )

    bench = subparsers.add_parser("bench", help="benchmark a local ONNX model")
    bench.add_argument("--model", required=True, help="path to a YOLO ONNX model")
    bench.add_argument("--runs", type=int, default=50)
    bench.add_argument("--backend", choices=["ncnn-vulkan", "openvino"], default=None)
    bench.add_argument(
        "--cache-dir",
        default=None,
        help="where to keep converted ncnn models (default: <model dir>/ncnn)",
    )

    status_parser = subparsers.add_parser(
        "status", help="show service, backend, model and live GPU status"
    )
    status_parser.add_argument("--unit", default=None, help="systemd unit name")
    status_parser.add_argument("--endpoint", default=None, help="ZMQ endpoint override")
    status_parser.add_argument("--model", default=None, help="ONNX model for --probe")
    status_parser.add_argument("--image", default=None, help="image for --probe")
    status_parser.add_argument("--crop", default=None, help="probe crop x1,y1,x2,y2")
    status_parser.add_argument("--runs", type=int, default=5, help="probe inference runs")
    status_parser.add_argument(
        "--timeout", type=int, default=10000, help="probe inference timeout in ms"
    )
    status_parser.add_argument(
        "--model-timeout", type=int, default=180000, help="model transfer timeout in ms"
    )
    status_parser.add_argument(
        "--sample-seconds", type=float, default=5.0, help="live CPU sample interval"
    )
    status_parser.add_argument("--quick", action="store_true", help="skip live CPU sampling")
    status_parser.add_argument("--probe", action="store_true", help="run an inference probe")
    status_parser.add_argument("--json", action="store_true", help="emit JSON")
    status_parser.add_argument("--check", action="store_true", help="exit 1 when unhealthy")
    status_parser.add_argument(
        "--require-model", action="store_true", help="fail --check when no model is loaded"
    )
    status_parser.add_argument("--quiet", action="store_true", help="suppress human output")

    probe_parser = subparsers.add_parser(
        "probe", help="run a local ZMQ protocol and inference probe"
    )
    probe_parser.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    probe_parser.add_argument("--model", required=True, help="path to the ONNX model")
    probe_parser.add_argument("--model-name", default=None)
    probe_parser.add_argument("--image", default=None)
    probe_parser.add_argument("--crop", default=None, help="crop x1,y1,x2,y2")
    probe_parser.add_argument("--runs", type=int, default=20)
    probe_parser.add_argument("--timeout", type=int, default=10000)
    probe_parser.add_argument("--model-timeout", type=int, default=180000)
    probe_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command in (None, "serve"):
        from .server import main as serve

        return serve()
    if args.command == "doctor":
        return _doctor()
    if args.command == "preflight":
        return _preflight()
    if args.command == "bench":
        return _bench(args)
    if args.command == "status":
        from .status import main as status_main

        return status_main(args)
    if args.command == "probe":
        from .probe import main as probe_main

        return probe_main(args)
    parser.print_help()
    return 2
