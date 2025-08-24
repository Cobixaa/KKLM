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

- NTT over 998244353 (power-of-two size):
```cpp
bool kllm::ntt_inplace(std::vector<std::uint32_t> &a, bool invert);
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

- Memory:
```cpp
std::unique_ptr<void, kllm::FreeDeleter> kllm::allocate_aligned_bytes(std::size_t size_bytes, std::size_t alignment);

template <typename T>
std::unique_ptr<T, kllm::FreeDeleter> kllm::allocate_aligned(std::size_t count, std::size_t alignment);

kllm::Status kllm::set_current_thread_affinity_status(int cpu_index);
bool kllm::set_current_thread_affinity(int cpu_index);
```

- IR, planner, and evaluation:
```cpp
struct kllm::Tensor { std::vector<float> values; };
struct kllm::Node { virtual ~Node(); virtual Tensor evaluate() = 0; };

struct kllm::GraphBuilder {
	static std::shared_ptr<Node> input(const std::vector<float> &values);
	static std::shared_ptr<Node> transform(const std::shared_ptr<Node> &in);
	static std::shared_ptr<Node> relu(const std::shared_ptr<Node> &in);
};

struct kllm::Planner {
	static std::shared_ptr<Node> plan(const std::shared_ptr<Node> &root);
};
```

- Quantization:
```cpp
struct kllm::QuantParams { float scale; };
kllm::QuantParams kllm::choose_symmetric_int8_scale(const float *data, std::size_t length);
void kllm::quantize_int8(const float *input, std::size_t length, int8_t *output, const QuantParams &params);
void kllm::dequantize_int8(const int8_t *input, std::size_t length, float scale, float *output);
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

### Performance Notes
- Prefer power-of-two lengths for transforms.
- Use `-march=native -O3`. Pin threads via `set_current_thread_affinity` for NUMA. On Linux/Android, affinity uses `sched_setaffinity()`; returns detailed `Status` via `set_current_thread_affinity_status`.
- Keep working sets within L1/L2; consider tiling at the call site.
- Fuse downstream pointwise ops with transforms (see IR planner) to reduce memory traffic.
- For int8 paths, pack/accumulate in int32 and dequantize late; batch operations for better cache locality.
- On aarch64, build with `-march=armv8-a+simd`; NEON kernels are enabled automatically.
- Set `kllm::set_parallel_threshold()` and `kllm::set_num_threads()` to tune parallel fused FWHT. Use `kllm::set_pin_threads(true)` to request worker pinning.

---

### License
You may use, copy, and modify this code for personal, educational, or research purposes only.
You may NOT sell or distribute this software or derivatives for commercial purposes without explicit permission.