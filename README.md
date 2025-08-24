## KLLM — CPU-first, Mobile-ready DL Primitives with Optional GPU (Header-only)

KLLM (Key-Light Large Model) is a C++17 header-only runtime of high-performance transform and sketch primitives with fused microkernels, a tiny IR + planner, streaming pipeline, and quantization to int8/int4.

- Header-only, zero external deps by default; builds on Linux and Android/Termux
- AVX2/NEON-optimized FWHT, scalar fallback
- Optional OpenCL GPU FWHT (best-effort): x86/ARM GPUs when available
- Streaming pipeline v2.1: Transform → Sketch → Route → Quantize with slab buffering
- Parallel sketch, routing, and blockwise quantization
- Tiny IR + planner that fuses Transform+Relu
- int8/int4 blockwise quantization, dequant helpers
- Robust status handling, thread pool, affinity (Linux)

---

### What’s new (v2.2)
- Parallel SketchEngine and RoutingEngine (across buckets)
- Parallel blockwise quantization (int8/int4)
- Pipeline buffer reuse to reduce allocations
- Optional OpenCL FWHT path with `set_enable_gpu(true)` and `-DKLLM_USE_OPENCL`
- Bench harness prints per-stage timings and supports `KLLM_GPU=1`

Performance snapshot (this environment):
- Pipeline v2.1 int8 1M: 26–30 ms (down from ~55 ms) depending on run
- FWHT 1M: ~8.2 ms CPU; fused FWHT-scale-add 1M: ~5.6 ms

Your mileage varies by CPU/GPU.

---

### Layout
- `kklm.h`: single public header
- `examples/main.cpp`: usage demo
- `bench/bench.cpp`: micro-benchmarks
- `test.cpp`: correctness tests

---

### Build (CPU-only)
Dependencies: clang++ (or g++), Linux or Android/Termux.

```bash
# x86-64
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I. examples/main.cpp -o kllm_demo
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I. bench/bench.cpp -o kllm_bench
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I. test.cpp -o kllm_test

# aarch64 (Termux)
clang++ -std=c++17 -O3 -march=armv8-a+simd -mtune=native -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I. examples/main.cpp -o kllm_demo
```

Run:
```bash
./kllm_demo
./kllm_bench
./kllm_test
```

---

### Optional GPU (OpenCL)
- Compile with `-DKLLM_USE_OPENCL` and link OpenCL (`-lOpenCL` on most distros)
- Enable at runtime: `kllm::set_enable_gpu(true)` or `KLLM_GPU=1` for bench
- Falls back to CPU if OpenCL platform/device is not found

Example build:
```bash
clang++ -std=c++17 -O3 -march=native -fPIC -Wall -Wextra -Wpedantic -Werror -I. \
  -DKLLM_USE_OPENCL examples/main.cpp -lOpenCL -o kllm_demo
```

Termux note: OpenCL availability varies by device/ROM. CPU path remains fully supported.

---

### Public API
Include the header:
```cpp
#include "kklm.h"
```

FWHT (in-place), input length must be power-of-two:
```cpp
kllm::fwht_inplace(ptr, n);
kllm::fwht_inplace_parallel(ptr, n, pool);
kllm::fwht_inplace_inverse(ptr, n);
```

v2.1 pipeline helpers:
```cpp
kllm::PipelineTelemetry t{};
std::vector<int8_t> q8; std::vector<float> scales;
auto st = kllm::run_pipeline_v21_to_int8(input, sketch_size, q8, scales, t, kllm::PointwiseOp::kRelu);
```

Quantization:
```cpp
kllm::BlockwiseQuantConfig qcfg; qcfg.block_size = 64;
std::vector<int8_t> q; std::vector<float> sc;
kllm::blockwise_quantize_int8(x.data(), x.size(), qcfg, q, sc);
```

Threading/Config:
```cpp
kllm::set_num_threads(8);
kllm::set_parallel_threshold(1<<14);
kllm::set_large_slab_bytes(1024*1024);
kllm::set_enable_gpu(true); // requires OpenCL build
```

---

### Benchmarks (sample)
```text
FWHT 1M floats: ~8.2 ms
FWHT(par,4) 1M floats: ~3.9–4.5 ms
Fused FWHT-scale-add 1M: ~5.6 ms
Pipeline v2.1 int8 1M: 26–30 ms, slabs=4 (stage1 dominates)
```

Tips:
- Increase `set_large_slab_bytes(1<<20)` and set `set_num_threads(6–12)`
- Keep `routing_bucket_size` near L2-sized tiles (default 256)

---

### Termux notes
- Some Android ROMs restrict CPU affinity. Use `set_current_thread_affinity_status()` to inspect errors.
- OpenCL may not be available on many devices; CPU path is optimized for NEON.

---

### License
You may use, copy, and modify this code for personal, educational, or research purposes only.
You may NOT sell or distribute this software or derivatives for commercial purposes without explicit permission.