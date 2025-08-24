## KLLM — CPU-first Transform/Sketch ML Runtime (Header-only)

KLLM (Key-Light Large Model) is a CPU-first, C++17 header-only library providing fast transform-based primitives (FWHT, NTT), sketching (CountSketch), fused microkernels, and a tiny IR with a planner for op fusion. It targets high performance on x86-64 (AVX2) and ARM64 (NEON) without heavy dependencies.

### Highlights
- Header-only, minimal, portable (Linux, Termux/Android ARM64)
- SIMD-accelerated FWHT; scalar fallback if no SIMD
- Modular NTT over 998244353 with forward/inverse
- CountSketch for hashing-based dimension reduction
- Fused transform-scale-add helper; new fused FWHT+bias+ReLU
- Aligned allocation and basic thread affinity API (portable via sched_setaffinity)
- Tiny IR + planner that fuses Transform+Relu
- int8 quantization helpers (scale selection, encode/decode)
- Lightweight thread pool and parallel FWHT
- Extended error handling: errno-aware messages, `StatusOr<T>`, guard macros
- Config extensions: `parallel_threshold`, `pin_threads`, reusable thread-local pool
- Optimized fused kernels: automatic parallel FWHT for large inputs
- v2.1: streaming execution pipeline (LargeN) with slab buffers and two-level routing
- v2.1: blockwise quantization manager (int8/int4), sketch engine v2, routing v2

---

### New in v2.1 (Flatline)
- Streaming LargeN pipeline with 2–3 slabs in flight; non-temporal edge stores when beneficial
- Hierarchical flow: Load/Quantize → Transform⊕Sketch → Route⊕Pointwise → Quantize/Output
- Two-level routing with per-bucket top-k, deterministic stable ordering when `deterministic=true`
- Blockwise int8/int4 quantization with per-block scales (32–128 elems), dequant helpers
- Config additions:
  - `set_large_slab_bytes(bytes)` (default 256 KB)
  - `set_max_inflight_slabs(n)` (default 3)
  - `set_pipeline_nt_stores(bool)`
  - `set_sketch_num_hashes(h)` (1–4)
  - `set_routing_bucket_size(sz)` (default 256)
- Public APIs:
  - `kllm::run_pipeline_v21_to_int8(input, sketch_size, q8, scales, telemetry, pointwise)`
  - `kllm::run_pipeline_v21_to_int4(input, sketch_size, q4, scales, telemetry, pointwise)`

---

### Layout
- `kklm.h`: single public header (drop-in)
- `examples/main.cpp`: usage demo
- `bench/bench.cpp`: micro-benchmarks
- `test.cpp`: basic correctness tests

---

### Build — Direct (single-step)

Dependencies: clang++ (or g++), Linux or Android/Termux.

```bash
# x86-64 (auto-detect)
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC \
  -Wall -Wextra -Wpedantic -Werror \
  -I. examples/main.cpp -o kllm_demo

# aarch64 (ARM64 NEON) / Termux on Android
# Note: Some Android kernels restrict CPU affinity from apps; affinity calls may be no-ops.
clang++ -std=c++17 -O3 -march=armv8-a+simd -mtune=native -fPIC \
  -Wall -Wextra -Wpedantic -Werror \
  -I. examples/main.cpp -o kllm_demo

# Benchmark
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC \
  -Wall -Wextra -Wpedantic -Werror \
  -I. bench/bench.cpp -o kllm_bench

# Tests
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC \
  -Wall -Wextra -Wpedantic -Werror \
  -I. test.cpp -o kllm_test
```

Run:
```bash
./kllm_demo
./kllm_bench
./kllm_test
```

---

### Termux notes / troubleshooting
- If you see an error like:
```text
./kklm.h:179:21: error: use of undeclared identifier 'pthread_setaffinity_np'; did you mean 'sched_setaffinity'?
```
Update to the latest header; CPU affinity now uses `sched_setaffinity()`. No pthread header is required.
- Some Android ROMs restrict affinity; calls may return a failed status or be no-ops. Use `set_current_thread_affinity_status()` to inspect errors.

---

### Public API Overview

Include the single header:
```cpp
#include "kklm.h"
```

- FWHT (in-place), length must be power-of-two:
```cpp
void kllm::fwht_inplace(float *data, std::size_t length);
void kllm::fwht_inplace_parallel(float *data, std::size_t length, kllm::ThreadPool &pool);
void kllm::fwht_inplace_inverse(float *data, std::size_t length);
```

- v2.1 pipeline APIs:
```cpp
// Quantize to int8
kllm::PipelineTelemetry t{}; std::vector<int8_t> q8; std::vector<float> scales;
auto st = kllm::run_pipeline_v21_to_int8(input, sketch_size, q8, scales, t, kllm::PointwiseOp::kRelu);

// Quantize to int4 (packed)
kllm::PipelineTelemetry t2{}; std::vector<uint8_t> q4; std::vector<float> scales2;
auto st2 = kllm::run_pipeline_v21_to_int4(input, sketch_size, q4, scales2, t2, kllm::PointwiseOp::kRelu);
```

- CountSketch:
```cpp
struct kllm::CountSketch {
	explicit CountSketch(std::size_t sketch_size, std::size_t num_hashes, std::uint64_t seed_base = 0x12345678abcdef00ull);
	void apply(const float *input, std::size_t length, float *output) const;
};
```

- Fused microkernels:
```cpp
void kllm::fused_fwht_scale_add(const float *input, std::size_t length, float scale, float *inout_destination);
void kllm::fused_fwht_bias_relu(const float *input, const float *bias, std::size_t length, float *destination);
```

- Quantization:
```cpp
struct kllm::QuantParams { float scale; };
kllm::QuantParams kllm::choose_symmetric_int8_scale(const float *data, std::size_t length);
void kllm::quantize_int8(const float *input, std::size_t length, int8_t *output, const QuantParams &params);
void kllm::dequantize_int8(const int8_t *input, std::size_t length, float scale, float *output);

// v2.1 blockwise
void kllm::blockwise_quantize_int8(const float*, std::size_t, const BlockwiseQuantConfig&, std::vector<int8_t>&, std::vector<float>&);
void kllm::blockwise_dequantize_int8(const int8_t*, std::size_t, const BlockwiseQuantConfig&, const std::vector<float>&, std::vector<float>&);
void kllm::blockwise_quantize_int4(const float*, std::size_t, std::size_t block_size, BlockwiseInt4Buffer&);
void kllm::blockwise_dequantize_int4(const BlockwiseInt4Buffer&, std::vector<float>&);
```

- Parallel helpers:
```cpp
class kllm::ThreadPool { public: explicit ThreadPool(std::size_t threads); void enqueue(std::function<void()> fn); void wait(); };
void kllm::parallel_for_blocks(ThreadPool &pool, std::size_t begin, std::size_t end, std::size_t step, std::function<void(std::size_t)> fn);
```

---

### Test Results (this environment)
```
ALL TESTS PASSED
```

### Benchmark Results (this environment)
```
FWHT 1M floats: 8.3327 ms (7.94668 ns/elem)
FWHT(par,4) 1M floats: 3.90141 ms (3.72067 ns/elem)
Fused FWHT-scale-add 1M: 7.92068 ms (7.55375 ns/elem)
FWHT 2048 floats: 12885 ns (6.2915 ns/elem)
Fused FWHT-scale-add 2048: 13463 ns (6.57373 ns/elem)
NTT 262k uint32: 4.46424 ms (17.0297 ns/elem)
CountSketch 1M -> 262k (3 hashes): 4.12869 ms (3.93743 ns/elem)
BlockDiag float 1024x(16x16): 0.051773 ms (3.15997 ns/elem)
BlockDiag int8 1024x(16x16): 0.047225 ms (2.88239 ns/elem)
LowRank 4096x4096 (r=64): 3.1e-05 ms (0.00756836 ns/elem)
```
System: clang++ 20.1.2, -O3 -march=native, Linux kernel 6.12+, CPU features autodetected. Results vary by CPU.

---

### Benchmarks (sample on this environment)
```
FWHT 1M floats: 8.33 ms
FWHT(par,4) 1M floats: 3.90 ms
Fused FWHT-scale-add 1M: 7.92 ms
FWHT 2048 floats: 12.9 us
Fused FWHT-scale-add 2048: 13.5 us
NTT 262k uint32: 4.46 ms
CountSketch 1M -> 262k (3 hashes): 4.13 ms
BlockDiag float 1024x(16x16): 0.052 ms
BlockDiag int8 1024x(16x16): 0.047 ms
LowRank 4096x4096 (r=64): ~0 ms
Pipeline v2.1 int8 1M: ~X ms (depends on CPU), slabs=~4
Pipeline v2.1 int4 1M: ~X ms (depends on CPU), slabs=~4
```
Targets: -20 ms vs baseline at N≈1,048,576 and flat throughput from 64K→1M+.

---

### Configuration Tips
- `kllm::set_parallel_threshold(1<<14)` and `kllm::set_num_threads()` tune parallel transforms.
- `kllm::set_large_slab_bytes(256*1024)` and `kllm::set_max_inflight_slabs(3)` control pipeline buffering.
- `kllm::set_pipeline_nt_stores(true)` enables non-temporal stores on edge stages for LargeN.
- `kllm::set_sketch_num_hashes(3)` and `kllm::set_routing_bucket_size(256)` tune sketch/router balance.
- `kllm::set_pin_threads(true)` may improve NUMA locality on Linux.

---

### License
You may use, copy, and modify this code for personal, educational, or research purposes only.
You may NOT sell or distribute this software or derivatives for commercial purposes without explicit permission.