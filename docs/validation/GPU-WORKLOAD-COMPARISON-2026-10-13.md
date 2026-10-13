# GPU choice for the actual Path B workload — 2026-10-13

The earlier A40 preference based mainly on hourly price/availability is revised.
L40 is the stronger first alternative to evaluate for the current FP32 workload;
A40 is a cheaper-hour fallback, not an established speed or cost improvement.
This is an engineering recommendation, not measured cross-GPU performance.
No settings, runner, frozen package or paid resources changed in this review.

## Current source and measured evidence

`src/model.py::load_model` pins original FP32 weights, eager attention/experts,
no custom kernels, deterministic algorithms and TF32 disabled. `measure_input`
uses two full forwards (capture and native cleanup) plus eight layer15 replays.
At layer15 there is no downstream decoder layer to replay. Tensor FP16/BF16/FP8
and TF32 throughput is not the arithmetic mode currently selected.

The historical L40S paired diagnostic measured 4k/layer15 optimized collector
mean elapsed 1.9014 seconds per window over three pairs. It also records substantial
startup, cache verification, load, native checks and teardown overhead. It is not
a kernel profile, a representative benchmark or complete-workflow feasibility.
No equivalent A40/L40/H100 workload measurement exists here.

## Manufacturer peak specifications (not runtime predictions)

| GPU | Native FP32 TFLOPS | Memory bandwidth GB/s |
|---|---:|---:|
| A40 | 37.4 | 696 |
| L40 | 90.5 | 864 |
| L40S | 91.6 | 864 |
| H100 SXM | 67 | 3350 |

Primary sources:
- [A40 datasheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf)
- [L40 datasheet](https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf)
- [L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [PyTorch FP32/TF32 semantics](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)

L40 is marketed for visual computing but includes fourth-generation Tensor Cores.
Its native FP32 and bandwidth peaks are almost identical to L40S. Its lower tensor
throughput matters to other arithmetic modes; peak similarity still does not prove
equal timings for our kernel sizes, cache behavior, host CPU or memory access.
H100's greater bandwidth may help bandwidth-bound portions, but its native FP32
peak is lower than L40S. No bottleneck classification is established without profiling.

## Cost break-even, using the live-read snapshot

Secure hourly prices from `alternative-gpu-catalog-2026-10-13.json`, read at
2026-10-13T03:59:00Z: A40 .59, L40 .82, L40S 1.09, H100 SXM 3.99 USD/hour.
GPU-only cost = hourly price × measured time. Before migration/storage/overheads:
- A40 must be less than 1.847× slower than L40S to lower compute cost.
- L40 must be less than 1.329× slower; equal runtime saves about 24.8%.
- H100 SXM must be more than 3.661× faster to lower compute cost.

For illustration only, a perfectly native-FP32-compute-bound peak-scaling model
would make A40 91.6/37.4 = 2.449× slower, giving about 1.326× GPU cost despite its
lower hourly price. This is a sensitivity example, not a predicted runtime.

## Next evidence and operational limits

Use synchronized identical 4k cases with the same arithmetic, native checks,
warmup, timing/persistence and per-case logs. Compare dollars per validated window
and all-phase cost, not hourly rate or advertised tensor TFLOPS alone. Device or
arithmetic changes require fresh scoped native verification at frozen tolerances.

The current runner still binds L40S and the cached US-MO-1 volume. L40/A40 stock
in the catalog snapshot is in other regions. Cache preparation/migration has an
unreconciled cost and is not included in the authorized $.20 single diagnostic.
No automatic deployment, region migration, dtype change or multi-GPU benchmark
is authorized by this recommendation.
