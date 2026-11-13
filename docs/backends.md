# Backends

`FRIGATE_ZMQ_BACKEND` selects how the model runs. Both backends implement the same wire
protocol and the same YOLO post-processing, so Frigate cannot tell them apart.

| Backend | Device | Model format | When to use |
| --- | --- | --- | --- |
| `openvino` | CPU | ONNX (compiled by OpenVINO) | no GPU, Intel Arc/CPU, fallback |
| `ncnn-vulkan` | GPU via Vulkan | ONNX converted to ncnn by pnnx | AMD iGPU (this project's target), Intel iGPU |

## `openvino`

- Uses the OpenVINO CPU plugin with `PERFORMANCE_HINT=THROUGHPUT`, `NUM_STREAMS=1`.
- `FRIGATE_ZMQ_CPU_THREADS` controls `INFERENCE_NUM_THREADS`. On an 8-core Zen2, 6 threads gave
  the best latency (53–54 ms); 8 and 12 threads were not faster and doubled CPU use.
- Requires the optional dependency: `pip install 'frigate-igpu-detector[openvino]'`.
- This is the only backend that works without a GPU, and the automatic fallback target.

## `ncnn-vulkan`

Pipeline:

```text
Frigate ──ONNX bytes──►  models/<name>.onnx
                              │
                              │ pnnx (once per content hash)
                              ▼
                     models/ncnn/<name>-<sha12>.ncnn.param|bin
                              │
                              │ ncnn.Net + Vulkan (fp16)
                              ▼
                        GPU inference
```

Details:

- The conversion happens at model-load time and takes about 2 seconds for YOLOv9-m (80 MB).
  Frigate's model transfer timeout must accommodate it; the default is generous enough.
- Cache files are keyed by the ONNX SHA-256, so replacing a model re-converts it and prunes the
  previous version of the same name.
- `pnnx` is invoked as an external binary. The release binary is preferred; the pip wheel's
  console script imports PyTorch and is only used if no real binary exists.
- fp16 packed/storage/arithmetic are enabled. On RADV Renoir the output matches the OpenVINO
  fp32 backend closely (scores within ~0.02, identical detection counts on real frames).
- `FRIGATE_ZMQ_NCNN_THREADS=1` is intentional. ncnn's CPU worker threads spin while the GPU
  works: 6 threads cost ~33 ms CPU per inference, 1 thread costs ~3 ms, with no latency gain.
- The first inference after loading compiles shaders. The server runs one warmup inference so
  Frigate's 1000 ms request timeout is not hit by the first real request.
- A warmup longer than 5 seconds is treated as a software rasteriser (llvmpipe) and raises, so
  the server falls back to OpenVINO instead of silently running at ~1 s per inference.

## Fallback behaviour

1. If `FRIGATE_ZMQ_BACKEND=ncnn-vulkan` and Vulkan/GPU initialisation fails, the server logs the
   reason and starts on OpenVINO CPU.
2. If the GPU is fine but a specific model fails to convert or load, that model is retried on
   OpenVINO CPU.
3. If OpenVINO is not installed either, model loading fails and Frigate is told the model is not
   ready; the error is in the log.

The startup log always states what is actually running:

```text
Inference backend: ncnn Vulkan, devices=2, gpu=0, threads=1, cache=/var/lib/frigate-detector/models/ncnn
Loaded ONNX model=yolov9-m-320.onnx ... backend=ncnn-vulkan gpu=0 threads=1 fp16=1 warmup=0.05s
```

or

```text
Vulkan/GPU initialisation failed; falling back to OpenVINO CPU
...
Inference backend: OpenVINO CPU, threads=6
```

## Measuring

```bash
fzd bench --model /path/to/model.onnx --backend ncnn-vulkan --runs 100
fzd bench --model /path/to/model.onnx --backend openvino --runs 30
```

`bench` prints a JSON summary (p50/p95/max latency and CPU per inference) that is useful in bug
reports.
