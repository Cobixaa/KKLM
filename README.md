## KLLM — CPU-first Transform/Sketch ML Runtime (Header-only)

KLLM (Key-Light Large Model) is a CPU-first, C++17 header-only library providing fast transform-based primitives (FWHT, NTT), sketching (CountSketch), fused microkernels, and a tiny IR with a planner for op fusion. It targets high performance on x86-64 (AVX2) and ARM64 (NEON) without heavy dependencies.

### Highlights
- Header-only, minimal, portable (Linux, Termux/Android ARM64)
- SIMD-accelerated FWHT; scalar fallback if no SIMD
- Modular NTT over 998244353 with forward/inverse
- CountSketch for hashing-based dimension reduction
- Fused transform-scale-add helper; new fused FWHT+bias+ReLU
- Aligned allocation and basic thread affinity API
- Tiny IR + planner that fuses Transform+Relu
- New: int8 quantization helpers (scale selection, encode/decode)
- New: lightweight thread pool (for future parallel transforms)

---

### Directory Layout
- `include/kllm/`: public headers
  - `kllm.h`: umbrella include
  - `utils.h`: misc utilities, prefetch, alignment macros
  - `fast_transform.h`: FWHT and inverse
  - `ntt.h`: iterative NTT mod 998244353
  - `sketch.h`: CountSketch
  - `fused.h`: fused FWHT + scale + add; fused FWHT+bias+ReLU
  - `memory.h`: aligned allocation and affinity
  - `ir.h`: minimal nodes, planner, and evaluate
  - `quant.h`: int8 quantization helpers
  - `parallel.h`: lightweight thread pool
- `examples/main.cpp`: demo exercising all APIs
- `bench/bench.cpp`: micro-benchmarks

---

### Build — Direct (no Make/CMake)

Dependencies: clang++ (or g++), Linux or Android/Termux.

Recommended flags (fast, warnings, native CPU):

```bash
# x86-64 AVX2 build
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC \
  -Wall -Wextra -Wpedantic \
  -Iinclude examples/main.cpp -o kllm_demo

# ARM64 NEON build (Termux/Android)
clang++ -std=c++17 -O3 -march=armv8-a+simd -mtune=native -fPIC \
  -Wall -Wextra -Wpedantic \
  -Iinclude examples/main.cpp -o kllm_demo

# Benchmark build
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC \
  -Wall -Wextra -Wpedantic \
  -Iinclude bench/bench.cpp -o kllm_bench
```

Run:
```bash
./kllm_demo
./kllm_bench
```

Notes:
- `-march=native` automatically enables AVX2/AVX512/NEON on the host. Use explicit `-mavx2` if needed.
- The library is header-only; just add `-Iinclude` to your project.

---

### Public API Overview

Include umbrella header:
```cpp
#include "kllm/kllm.h"
```

- FWHT (in-place), length must be power-of-two:
```cpp
void kllm::fwht_inplace(float *data, std::size_t length);
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

- Memory utilities:
```cpp
std::unique_ptr<void, kllm::FreeDeleter> kllm::allocate_aligned_bytes(std::size_t size_bytes, std::size_t alignment);

template <typename T>
std::unique_ptr<T, kllm::FreeDeleter> kllm::allocate_aligned(std::size_t count, std::size_t alignment);

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

- Quantization helpers:
```cpp
struct kllm::QuantParams { float scale; };
kllm::QuantParams kllm::choose_symmetric_int8_scale(const float *data, std::size_t length);
void kllm::quantize_int8(const float *input, std::size_t length, int8_t *output, const QuantParams &params);
void kllm::dequantize_int8(const int8_t *input, std::size_t length, float scale, float *output);
```

- Parallel helpers:
```cpp
class kllm::ThreadPool { public: explicit ThreadPool(std::size_t threads); void enqueue(std::function<void()> fn); };
```

---

### Benchmark Results (on this environment)
```
FWHT 1M floats: 8.21 ms
Fused FWHT-scale-add 1M: 10.08 ms
NTT 262k uint32: 8.13 ms
CountSketch 1M -> 262k (3 hashes): 4.30 ms
```
System: clang++ 20.1.2, -O3 -march=native, Linux kernel 6.12+, CPU features autodetected.

---

### Performance Notes
- Prefer power-of-two lengths for transforms.
- Use `-march=native -O3` and pin threads via `set_current_thread_affinity` for NUMA.
- Keep working sets within L1/L2; consider tiling at the call-site.
- Fuse downstream pointwise ops with transforms (see IR planner) to reduce memory traffic.
- For int8 paths, pack/accumulate in int32 and dequantize late; batch operations for better cache locality.

---

### License
MIT