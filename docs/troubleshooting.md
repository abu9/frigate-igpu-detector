# Troubleshooting

Start with:

```bash
fzd preflight
fzd doctor
fzd status
journalctl -u frigate-igpu-detector -n 100 --no-pager -o cat
```

## `ModuleNotFoundError: No module named 'ncnn'` (or `openvino`) after a restart

The service is probably using a virtualenv that no longer inherits the system site packages.
This commonly happens after running `python3 -m venv venv` again on an existing deployment:
that command rewrites `pyvenv.cfg` and resets `include-system-site-packages` to `false`.

Check and repair:

```bash
grep '^include-system-site-packages' \
  /opt/frigate-igpu-detector/venv/pyvenv.cfg

sed -i 's/^include-system-site-packages = .*/include-system-site-packages = true/' \
  /opt/frigate-igpu-detector/venv/pyvenv.cfg

systemctl restart frigate-igpu-detector
fzd preflight
```

If you install `ncnn` and OpenVINO inside the venv instead of using system packages, leave the
flag at `false` and make sure both extras are installed in that venv.

## Service is active but Frigate receives `no model is loaded`

Run:

```bash
fzd status
```

The `backend / model` section shows the active backend and whether a model is loaded. If both
`ncnn` and OpenVINO are unavailable, the current release exits at startup instead of listening
and returning zero results. Check the unit state:

```bash
systemctl status frigate-igpu-detector
journalctl -u frigate-igpu-detector -n 50 --no-pager -o cat
```

## Only `llvmpipe` is visible / warmup takes tens of seconds

`llvmpipe` is Mesa's CPU rasteriser. ncnn will happily use it and be ~20× slower.

Causes, in order of likelihood:

1. The container user is not allowed to open `/dev/dri/renderD128`:

   ```bash
   ls -l /dev/dri/
   # crw-rw---- 1 root render 226, 128 ... renderD128
   id
   getent group render
   ```

   Add the user to `render`, or pass the device with the container's render GID
   (`pct set 101 -dev0 /dev/dri/renderD128,gid=993`).

2. The node is visible as `nobody nogroup` in the container → GID mapping is wrong (legacy
   `lxc.mount.entry` on an unprivileged container). See [proxmox-lxc.md](proxmox-lxc.md).

3. `mesa-vulkan-drivers` is missing inside the container:

   ```bash
   apt install mesa-vulkan-drivers libvulkan1 vulkan-tools
   vulkaninfo --summary
   ```

The server refuses to run the Vulkan backend when warmup exceeds 5 seconds, and falls back to
OpenVINO with an explanatory log line.

## `Address family not supported by protocol` and `status=6/ABRT`

systemd's `RestrictAddressFamilies` is blocking a family ZeroMQ needs for name resolution.
Include `AF_NETLINK`:

```ini
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK
```

## All detections are zero after restarting the sidecar

Frigate asks for the model **once**, during its own startup. If the sidecar restarts later and
does not load a cached model, it answers every inference with zeros.

This server autoloads the newest `*.onnx` in `FRIGATE_ZMQ_MODEL_DIR` at startup, so the model
directory must be persistent. Check the log for:

```text
Auto-loaded cached model at startup: yolov9-m-320.onnx
```

If there is no cached model, restart Frigate (or reload its detector) so it sends the model
again.

## Model transfer times out

The first transfer includes pnnx conversion (~2 s for an 80 MB model). If Frigate's model
timeout is very low, raise it, or pre-convert by placing the ONNX in the model directory and
starting the sidecar once before starting Frigate.

## `pnnx not found`

- Install the release binary into `PATH`, or
- set `FRIGATE_ZMQ_PNNX=/path/to/pnnx`.

`fzd doctor` prints the resolved path. Do not rely on the pip console script unless PyTorch is
installed: the wheel's entry point imports `torch`, while the packaged ELF binary does not.

## `pnnx conversion failed`

The model uses an operator pnnx cannot map to ncnn. Options:

- re-export the model (simplify, static shapes, opset 12–17),
- run this model on the OpenVINO backend,
- open an issue with the pnnx log tail (the server logs the last 800 characters).

## Inference is fast but CPU is high

- On the Vulkan backend, set `FRIGATE_ZMQ_NCNN_THREADS=1`. Extra ncnn worker threads spin while
  the GPU works.
- If the log says `Inference backend: OpenVINO CPU`, the GPU path is not active. Look for the
  failure line above it.

## High p95 latency / spikes

The iGPU is shared with the host and other containers. Check:

```bash
cat /sys/class/drm/card1/device/gpu_busy_percent
```

Pause transcoding or other GPU workloads, or lower the detection load in Frigate (motion masks,
`detection_fps`).

## Frigate reports the detector as slow

- `inference_speed` around 47–56 ms for YOLOv9-m 320 is expected on a Vega 7.
- If `detection_fps` is several times `process_fps`, Frigate is creating multiple regions per
  frame (multiple tracked objects or noisy motion). That is a Frigate tuning issue, not a backend
  issue; see Frigate's motion tuning docs.

## Collecting a useful bug report

```bash
fzd preflight > preflight.txt 2>&1
fzd doctor > doctor.txt 2>&1
fzd status --json > status.json 2>&1
fzd bench --model /path/to/model.onnx --runs 100 > bench.txt 2>&1
journalctl -u frigate-igpu-detector -n 200 --no-pager -o cat > log.txt
vulkaninfo --summary > vulkan.txt 2>&1
```

Include the backend, GPU model, Mesa version, Frigate version and model family.
