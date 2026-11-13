"""Runtime status collection and rendering for frigate-igpu-detector."""

from __future__ import annotations

import grp
import hashlib
import json
import os
import pwd
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from . import __version__
from .cli import _deployed_environment, _parse_bind, _parse_proc_environ
from .probe import parse_crop, run_probe

DEFAULT_UNIT = "frigate-igpu-detector.service"
DEFAULT_IMAGE = "/var/lib/frigate-detector/healthcheck.jpg"


def _systemctl_show(unit: str, *properties: str) -> dict[str, str]:
    cmd = ["systemctl", "show", "--no-pager"]
    for prop in properties:
        cmd.extend(["-p", prop])
    cmd.append(unit)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _journal_lines(
    unit: str, pid: int | None = None, since: str | None = None, limit: int = 300
) -> list[str]:
    cmd = ["journalctl", "-n", str(limit), "-o", "cat", "--no-pager"]
    if since:
        cmd.extend(["--since", since])
    if pid:
        cmd.append(f"_PID={pid}")
    else:
        cmd.extend(["-u", unit])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return proc.stdout.splitlines()


def _parse_proc_status(path: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key.strip()] = value.strip()
    except OSError:
        pass
    return status


def _process_info(pid: int) -> dict[str, Any]:
    base = Path(f"/proc/{pid}")
    if not base.is_dir():
        return {}
    status = _parse_proc_status(base / "status")
    try:
        owner = pwd.getpwuid(base.stat().st_uid).pw_name
    except (KeyError, OSError):
        owner = str(status.get("Uid", "?"))
    groups: list[str] = []
    for gid in status.get("Groups", "").split():
        try:
            groups.append(f"{grp.getgrgid(int(gid)).gr_name}({gid})")
        except (KeyError, ValueError):
            groups.append(gid)
    try:
        argv = [
            part.decode("utf-8", "replace")
            for part in (base / "cmdline").read_bytes().split(b"\0")
            if part
        ]
    except OSError:
        argv = []
    environ: dict[str, str] = {}
    try:
        environ = {
            key: value
            for key, value in _parse_proc_environ((base / "environ").read_bytes()).items()
            if key.startswith("FRIGATE_ZMQ_") or key in ("HOME", "XDG_CACHE_HOME")
        }
    except OSError:
        pass
    rss_bytes = None
    if status.get("VmRSS"):
        try:
            rss_bytes = int(status["VmRSS"].split()[0]) * 1024
        except (ValueError, IndexError):
            pass
    return {
        "pid": pid,
        "user": owner,
        "groups": groups,
        "cmdline": argv,
        "rss_bytes": rss_bytes,
        "environment": environ,
    }


def _client_endpoint(endpoint: str | None) -> str | None:
    """Map wildcard bind addresses to a loopback endpoint for local probes."""
    if not endpoint:
        return endpoint
    parsed = _parse_bind(endpoint)
    if not parsed:
        return endpoint
    host, port = parsed
    if host in ("0.0.0.0", "::", "[::]", "*", ""):
        return f"tcp://127.0.0.1:{port}"
    return endpoint


def _listening(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    parsed = _parse_bind(endpoint)
    if not parsed:
        return False
    host, port = parsed
    if host in ("0.0.0.0", "::", "*", ""):
        host = "127.0.0.1"
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _gpu_busy_percent() -> int | None:
    for path in sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")):
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def _sha256_prefix(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return "?"
    return digest.hexdigest()[:length]


def _collect_models(model_dir: Path) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    if not model_dir.is_dir():
        return models
    for path in sorted(
        (candidate for candidate in model_dir.glob("*.onnx") if candidate.is_file()),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    ):
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        models.append(
            {
                "path": str(path),
                "name": path.name,
                "size_bytes": size,
                "sha256_prefix": _sha256_prefix(path),
            }
        )
    return models


def _collect_cache(model_dir: Path) -> list[dict[str, Any]]:
    cache_dir = model_dir / "ncnn"
    result: list[dict[str, Any]] = []
    if not cache_dir.is_dir():
        return result
    for path in sorted(cache_dir.glob("*.ncnn.*")):
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        result.append({"path": str(path), "name": path.name, "size_bytes": size})
    return result


def _parse_log_state(lines: list[str]) -> dict[str, Any]:
    backend = None
    loaded_model = None
    fallback = False
    for line in lines:
        match = re.search(r"Inference backend:\s*(.+?)(?:,|$)", line)
        if match:
            backend = match.group(1).strip()
        match = re.search(r"Loaded ONNX model=(\S+)", line)
        if match:
            loaded_model = match.group(1)
        match = re.search(r"Auto-loaded cached model at startup:\s*(\S+)", line)
        if match:
            loaded_model = match.group(1)
        match = re.search(r"Model transfer complete:\s*(\S+)", line)
        if match:
            loaded_model = match.group(1)
        if "falling back to OpenVINO" in line:
            fallback = True
    return {"backend": backend, "loaded_model": loaded_model, "fallback": fallback}


def _query_model_state(endpoint: str | None, name: str | None) -> dict[str, Any] | None:
    if not endpoint or not name:
        return None
    try:
        import zmq
    except Exception:
        return None
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDTIMEO, 2000)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    try:
        sock.connect(endpoint)
        sock.send_multipart(
            [json.dumps({"model_request": True, "model_name": name}).encode()]
        )
        parts = sock.recv_multipart()
        if len(parts) != 1:
            return None
        return json.loads(parts[0])
    except Exception:
        return None
    finally:
        sock.close(0)
        ctx.term()


def _live_sample(unit: str, pid: int, seconds: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cpu_cores": None,
        "gpu_busy_percent": _gpu_busy_percent(),
        "cgroup_memory_bytes": None,
        "tasks": None,
        "rss_bytes": None,
    }
    if pid:
        try:
            result["rss_bytes"] = int(
                _parse_proc_status(Path(f"/proc/{pid}/status"))
                .get("VmRSS", "0 kB")
                .split()[0]
            ) * 1024
        except (ValueError, IndexError):
            pass
    try:
        result["cgroup_memory_bytes"] = int(
            _systemctl_show(unit, "MemoryCurrent").get("MemoryCurrent", "")
        )
    except ValueError:
        pass
    try:
        result["tasks"] = int(
            _systemctl_show(unit, "TasksCurrent").get("TasksCurrent", "")
        )
    except ValueError:
        pass
    if seconds > 0:
        try:
            before = int(_systemctl_show(unit, "CPUUsageNSec").get("CPUUsageNSec", ""))
            time.sleep(seconds)
            after = int(_systemctl_show(unit, "CPUUsageNSec").get("CPUUsageNSec", ""))
            result["cpu_cores"] = round((after - before) / 1e9 / seconds, 3)
        except ValueError:
            pass
    return result


def _config(unit: str) -> dict[str, Any]:
    from . import server

    deployed, source = _deployed_environment(unit)
    cfg = {**os.environ, **deployed}
    bind_value = cfg.get("FRIGATE_ZMQ_BIND", server.BIND)
    model_dir = Path(cfg.get("FRIGATE_ZMQ_MODEL_DIR", str(server.MODEL_DIR)))
    return {
        "source": source,
        "backend": cfg.get("FRIGATE_ZMQ_BACKEND", server.BACKEND),
        "bind": bind_value,
        "model_dir": str(model_dir),
        "ncnn_threads": cfg.get("FRIGATE_ZMQ_NCNN_THREADS", str(server.NCNN_THREADS)),
        "cpu_threads": cfg.get("FRIGATE_ZMQ_CPU_THREADS", str(server.CPU_THREADS)),
        "endpoint": bind_value if bind_value.startswith("tcp://") else None,
    }


def collect_status(args: Any) -> dict[str, Any]:
    unit = args.unit or os.environ.get("FRIGATE_ZMQ_UNIT", DEFAULT_UNIT)
    service = _systemctl_show(
        unit,
        "ActiveState",
        "SubState",
        "MainPID",
        "ActiveEnterTimestamp",
        "UnitFileState",
        "FragmentPath",
        "MemoryCurrent",
        "CPUUsageNSec",
        "TasksCurrent",
    )
    try:
        pid = int(service.get("MainPID", "0") or 0)
    except ValueError:
        pid = 0
    config = _config(unit)
    endpoint = _client_endpoint(getattr(args, "endpoint", None) or config.get("endpoint"))
    model_dir = Path(config["model_dir"])

    log_lines = _journal_lines(unit, pid or None)
    log_state = _parse_log_state(log_lines)
    if not log_state["backend"]:
        log_state["backend"] = config["backend"] if service.get("ActiveState") == "active" else None

    proc = _process_info(pid) if pid else {}
    models = _collect_models(model_dir)
    model_state = _query_model_state(endpoint, log_state.get("loaded_model"))
    listening = _listening(endpoint)
    sample_seconds = 0.0 if getattr(args, "quick", False) else float(
        getattr(args, "sample_seconds", 5.0)
    )
    live = _live_sample(unit, pid, sample_seconds) if pid else {}

    error_lines = [
        line
        for line in _journal_lines(unit, pid or None, since="-30min", limit=500)
        if re.search(r"\b(ERROR|FATAL|Traceback|rejected)\b", line)
    ][-5:]

    probe_result = None
    probe_error = None
    if getattr(args, "probe", False):
        model_path = Path(args.model) if getattr(args, "model", None) else None
        if model_path is None and models:
            model_path = Path(models[0]["path"])
        image = getattr(args, "image", None)
        if image is None and Path(DEFAULT_IMAGE).is_file():
            image = DEFAULT_IMAGE
        if model_path is None:
            probe_error = "no ONNX model found for probe"
        else:
            crop = parse_crop(getattr(args, "crop", None))
            probe_result = run_probe(
                endpoint=endpoint or "tcp://127.0.0.1:5555",
                model_path=model_path,
                image_path=Path(image) if image else None,
                crop=crop,
                runs=getattr(args, "runs", 5),
                timeout_ms=getattr(args, "timeout", 10000),
                model_timeout_ms=getattr(args, "model_timeout", 180000),
            )

    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add_check(
        "service",
        service.get("ActiveState") == "active",
        f"{service.get('ActiveState', 'unknown')} ({service.get('SubState', 'unknown')})",
    )
    add_check("pid", pid > 0, str(pid))
    add_check("listening", listening, endpoint or "no tcp endpoint")
    add_check(
        "backend",
        bool(log_state.get("backend")),
        str(log_state.get("backend") or "unknown"),
    )
    if getattr(args, "require_model", False):
        loaded = bool(model_state and model_state.get("model_loaded"))
        add_check("model", loaded, log_state.get("loaded_model") or "not loaded")
    if getattr(args, "probe", False):
        if probe_error:
            add_check("probe", False, probe_error)
        else:
            add_check(
                "probe",
                bool(probe_result and probe_result.get("ok")),
                str((probe_result or {}).get("error") or "ok"),
            )

    healthy = all(check["ok"] for check in checks)
    return {
        "version": __version__,
        "unit": unit,
        "service": service,
        "process": proc,
        "config": config,
        "backend": log_state,
        "models": models,
        "cache": _collect_cache(model_dir),
        "model_state": model_state,
        "listening": listening,
        "live": live,
        "errors": error_lines,
        "probe": probe_result,
        "probe_error": probe_error,
        "checks": checks,
        "healthy": healthy,
    }


def _render_environment(env: dict[str, str], config: dict[str, Any]) -> None:
    if env:
        for key in sorted(env):
            if key.startswith("FRIGATE_ZMQ_") or key in ("HOME", "XDG_CACHE_HOME"):
                print(f"{key}={env[key]}")
    else:
        print(f"config source: {config.get('source')}")
        print(f"FRIGATE_ZMQ_BACKEND={config.get('backend')}")
        print(f"FRIGATE_ZMQ_BIND={config.get('bind')}")
        print(f"FRIGATE_ZMQ_MODEL_DIR={config.get('model_dir')}")


def render_status(data: dict[str, Any], show_checks: bool = False) -> None:
    service = data.get("service", {})
    process = data.get("process", {})
    print("===== service =====")
    print(f"{service.get('ActiveState', 'unknown')} ({service.get('SubState', 'unknown')})")
    if service.get("ActiveEnterTimestamp"):
        print(f"since: {service['ActiveEnterTimestamp']}")
    print(f"main pid: {service.get('MainPID', '0')}")
    print(f"unit file: {service.get('UnitFileState', 'unknown')}")

    print("\n===== process =====")
    if process:
        print(
            f"{process.get('user')} pid={process.get('pid')} "
            f"rss={_fmt_bytes(process.get('rss_bytes'))}"
        )
        print(f"groups: {' '.join(process.get('groups') or []) or 'n/a'}")
        print(f"cmdline: {' '.join(process.get('cmdline') or [])}")
    else:
        print("(not running)")

    print("\n===== effective environment =====")
    _render_environment(process.get("environment") or {}, data.get("config", {}))

    backend = data.get("backend", {})
    model_state = data.get("model_state") or {}
    print("\n===== backend / model =====")
    print(f"configured: {data.get('config', {}).get('backend')}")
    print(f"active:     {backend.get('backend') or 'unknown'}")
    print(f"fallback:   {bool(backend.get('fallback'))}")
    print(f"model:      {backend.get('loaded_model') or 'none'}")
    if model_state:
        print(
            "loaded:     "
            f"{model_state.get('model_loaded')} "
            f"(available={model_state.get('model_available')})"
        )
    else:
        print("loaded:     unknown")

    print("\n===== model / cache =====")
    if data.get("models"):
        for model in data["models"]:
            print(
                f"{model['sha256_prefix']}  {_fmt_bytes(model.get('size_bytes'))}  "
                f"{model['name']}"
            )
    else:
        print("(no ONNX model)")
    for item in data.get("cache") or []:
        print(f"{_fmt_bytes(item.get('size_bytes'))}  {item['name']}")

    live = data.get("live") or {}
    print("\n===== live load =====")
    cpu_cores = live.get("cpu_cores")
    gpu_busy = live.get("gpu_busy_percent")
    print(f"cpu: {cpu_cores if cpu_cores is not None else 'n/a'} cores")
    print(f"gpu_busy: {gpu_busy if gpu_busy is not None else 'n/a'}%")
    print(f"cgroup memory: {_fmt_bytes(live.get('cgroup_memory_bytes'))}")
    print(f"VmRSS: {_fmt_bytes(live.get('rss_bytes'))}")

    print("\n===== recent errors (30 min) =====")
    if data.get("errors"):
        for line in data["errors"]:
            print(line)
    else:
        print("(none)")

    if data.get("probe") is not None or data.get("probe_error"):
        print("\n===== end-to-end probe =====")
        if data.get("probe_error"):
            print(f"ERROR: {data['probe_error']}")
        else:
            result = data["probe"]
            if result.get("before") is not None:
                print(f"before: {json.dumps(result['before'])}")
            if result.get("after") is not None:
                print(f"after:  {json.dumps(result['after'])}")
            if result.get("detections"):
                print("detections (last run):")
                for det in result["detections"]:
                    box = ",".join(f"{value:.3f}" for value in det["box"])
                    print(f"  {det['label']:16s} score={det['score']:.3f} box=({box})")
            elif result.get("runs"):
                print("detections (last run):\n  (none)")
            if result.get("latency_ms"):
                print(f"SUMMARY {json.dumps(result, separators=(',', ':'))}")
            if not result.get("ok"):
                print(f"ERROR: {result.get('error')}")

    if show_checks:
        print("\n===== checks =====")
        for check in data.get("checks", []):
            marker = "OK" if check["ok"] else "FAIL"
            print(f"[{marker:4s}] {check['name']}: {check['detail']}")


def main(args: Any) -> int:
    try:
        data = collect_status(args)
    except Exception as exc:
        print(f"status failed: {type(exc).__name__}: {exc}")
        return 2
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif not getattr(args, "quiet", False):
        render_status(data, show_checks=getattr(args, "check", False))
    if getattr(args, "check", False):
        return 0 if data.get("healthy") else 1
    return 0


if __name__ == "__main__":
    raise SystemExit("Use `fzd status` instead of running this module directly.")
