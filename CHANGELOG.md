# Changelog

All notable changes to this project are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses semantic versioning.

## [0.1.2] - 2026-11-13

### Fixed

- Running `fzd` without a subcommand now prints help instead of accidentally starting a
  second server and failing with `Address already in use`.

## [0.1.1] - 2026-11-13

### Fixed

- Exit with a clear error when neither ncnn nor OpenVINO is usable, instead of staying active
  and rejecting every inference request with `no model is loaded`.
- Detect missing system site packages at startup and during `fzd preflight`.
- Clean up a partially initialised Vulkan instance when GPU setup fails.

### Added

- `fzd preflight` for systemd `ExecStartPre` and manual dependency checks.
- `fzd status` with human-readable output, `--json`, `--check`, `--probe` and `--require-model`.
- `fzd probe` for standalone ZMQ model-transfer and inference testing.
- systemd start limits and preflight integration in the unit template.

### Changed

- `fzd` can be installed on `PATH` via a symlink; the installation guide shows the step.
- Documentation covers `include-system-site-packages`, venv recreation pitfalls and the new CLI.

## [0.1.0] - 2026-11-13

### Added

- ZMQ remote detector server compatible with Frigate 0.18.x (`model_request`,
  `model_data`, inference, 20×6 float32 replies).
- `ncnn-vulkan` backend: fp16 Vulkan inference on AMD/Intel GPUs, ONNX → ncnn conversion via
  pnnx, content-hash cache, shader warmup, automatic fallback to OpenVINO.
- `openvino` CPU backend (fallback and non-GPU deployments).
- Startup autoload of the newest cached model, so sidecar restarts do not break a running
  Frigate instance.
- `fzd` CLI with `serve`, `doctor` and `bench` subcommands.
- Proxmox LXC deployment guide, backend notes and troubleshooting documentation.
- English and Simplified Chinese README.
- systemd unit template and Frigate detector configuration example.
