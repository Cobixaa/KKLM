## KLLM — CPU-first Transform/Sketch ML Runtime (Header-only)

KLLM (Key-Light Large Model) is a CPU-first, C++17 header-only library providing fast transform-based primitives (FWHT, NTT), sketching (CountSketch), fused microkernels, and a tiny IR with a planner for op fusion. It targets high performance on x86-64 (AVX2) and ARM64 (NEON) without heavy dependencies.

### Highlights
- Header-only, minimal, portable (Linux, Termux/Android ARM64)
- SIMD-accelerated FWHT; scalar fallback if no SIMD
- Modular NTT over 998244353 with forward/inverse
- CountSketch for hashing-based dimension reduction
- Fused transform-scale-add helper
- Aligned allocation and basic thread affinity API
- Tiny IR + planner that fuses Transform+Relu

---

### Directory Layout
- `include/kllm/`: public headers
  - `kllm.h`: umbrella include
  - `utils.h`: misc utilities, prefetch, alignment macros
  - `fast_transform.h`: FWHT and inverse
  - `ntt.h`: iterative NTT mod 998244353
  - `sketch.h`: CountSketch
  - `fused.h`: fused FWHT + scale + add
  - `memory.h`: aligned allocation and affinity
  - `ir.h`: minimal nodes, planner, and evaluate
- `examples/main.cpp`: demo exercising all APIs

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
```

Run:
```bash
./kllm_demo
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

- Fused microkernel helper:
```cpp
void kllm::fused_fwht_scale_add(const float *input, std::size_t length, float scale, float *inout_destination);
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

---

### Usage Example
```cpp
#include "kllm/kllm.h"
#include <vector>

int main() {
	std::vector<float> x = {1,2,3,4,5,6,7,8};
	kllm::fwht_inplace(x.data(), x.size());
	kllm::fwht_inplace_inverse(x.data(), x.size());

	kllm::CountSketch cs(8, 3);
	std::vector<float> y(8, 0.0f);
	cs.apply(x.data(), x.size(), y.data());

	auto n0 = kllm::GraphBuilder::input(x);
	auto n1 = kllm::GraphBuilder::transform(n0);
	auto n2 = kllm::GraphBuilder::relu(n1);
	auto planned = kllm::Planner::plan(n2);
	auto out = planned->evaluate();
	return static_cast<int>(out.values.size());
}
```

---

### Performance Notes
- Prefer power-of-two lengths for transforms.
- Use `-march=native -O3` and pin threads via `set_current_thread_affinity` for NUMA.
- Keep working sets within L1/L2; consider tiling at the call-site.
- Fuse downstream pointwise ops with transforms (see IR planner) to reduce memory traffic.

---

### License
MIT