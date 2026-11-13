# Changelog

All notable changes to this project are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses semantic versioning.

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
