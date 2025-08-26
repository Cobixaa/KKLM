## KLLM — CPU-first, Mobile-ready DL Primitives (Header-only, CPU-only)

KLLM (Key-Light Large Model) is a C++17 header-only runtime of high-performance transform and sketch primitives with fused microkernels, a tiny IR + planner, streaming pipeline, and quantization to int8/int4.

- Header-only, zero external deps by default; builds on Linux and Android/Termux
- AVX2/NEON-optimized FWHT, scalar fallback
- GPU support removed: streamlined CPU-only path for maximum portability
- Streaming pipeline v2.1: Transform → Sketch → Route → Quantize with slab buffering
- Parallel sketch, routing, and blockwise quantization
- Tiny IR + planner that fuses Transform+Relu
- int8/int4 blockwise quantization, dequant helpers
- Robust status handling, thread pool, affinity (Linux)

---

### What’s new (v2.2 CPU)
- Parallel SketchEngine and RoutingEngine (across buckets)
- Parallel blockwise quantization (int8/int4)
- Pipeline buffer reuse to reduce allocations
- GPU code paths removed; simpler build and predictable performance on CPU
- Autograd memory fix: safer graph ownership (parents now held as shared_ptr) to avoid leaks and UAF
- New runtime knobs: `enable_simd`, `enable_matmul_blocked`, `matmul_block_{m,n,k}`, `release_tls_fused_buffers`

Performance snapshot (this environment):
- FWHT 1M: ~2.64–5.22 ms (parallel vs single)
- Fused FWHT-scale-add 1M: ~6.5 ms
- Pipeline v2.1 int8 1M: ~24–30 ms
- Pipeline v2.1 int4 1M: ~23–28 ms

Your mileage varies by CPU.

---

### Layout
- `kklm.h`: single public header
- `examples/main.cpp`: usage demo
- `examples/miko.cpp`: toy self-play chess learner with emoji board and save/load
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
  -I. examples/miko.cpp -o miko
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I. bench/bench.cpp -o kllm_bench
clang++ -std=c++17 -O3 -march=native -mtune=native -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I. test.cpp -o kllm_test

# aarch64 (Termux)
clang++ -std=c++17 -O3 -march=armv8-a+simd -mtune=native -fPIC -Wall -Wextra -Wpedantic -Werror \
  -I. examples/miko.cpp -o miko
```

Run:
```bash
./kllm_demo
./kllm_bench
./kllm_test
./miko
```

---

### GPU
Removed. CPU-only.

---

### Public API
Include the header:
```cpp
#include "kklm.h"
```

Config/tuning (CPU):
```cpp
kllm::set_num_threads(8);                  // threads (0 uses hardware_concurrency)
kllm::set_parallel_threshold(1<<14);       // size threshold for parallel paths
kllm::set_large_slab_bytes(1024*1024);     // pipeline slab size
// Direct config access (advanced):
kllm::global_config().enable_simd = true;            // runtime SIMD on/off
kllm::global_config().enable_matmul_blocked = true;  // blocked GEMM on/off
kllm::global_config().matmul_block_m = 64;           // tile sizes
kllm::global_config().matmul_block_n = 128;
kllm::global_config().matmul_block_k = 128;
kllm::global_config().release_tls_fused_buffers = true; // free fused TLS buffers each call
```

Core transforms:
```cpp
// FWHT in-place; length must be power-of-two
kllm::fwht_inplace(ptr, n);
kllm::fwht_inplace_parallel(ptr, n, pool);
kllm::fwht_inplace_inverse(ptr, n);

// Fused microkernels
kllm::fused_fwht_scale_add(x, n, scale, y);
kllm::fused_fwht_bias_relu(x, bias, n, out);
```

Pipelines and quant:
```cpp
kllm::PipelineTelemetry T{};
std::vector<int8_t> q8; std::vector<float> scales;
// Transform -> Sketch -> Pointwise -> Route -> Quantize
kllm::run_pipeline_v21_to_int8(input, sketch_size, q8, scales, T, kllm::PointwiseOp::kRelu);
```

Sketch and routing:
```cpp
kllm::CountSketch cs(1<<12, 3);
cs.apply(x.data(), x.size(), y.data());
```

Rewards/metrics:
```cpp
float mse = kllm::reward_mse(pred, target);
float cos = kllm::reward_cosine_similarity(a, b);
float acc = kllm::reward_top1_accuracy(logits, labels, num_classes);
float f1  = kllm::reward_f1_binary(pred_labels, true_labels);
float bleu = kllm::reward_bleu_1_4(seq_pred, seq_ref);
float rouge = kllm::reward_rouge_l(seq_pred, seq_ref);
```

Threading/config (CPU):
```cpp
kllm::set_num_threads(8);
kllm::set_parallel_threshold(1<<14);
kllm::set_large_slab_bytes(1024*1024);
```

---

### nn/autograd quickstart
Build a tiny MLP classifier using the new minimal nn API:
```cpp
using namespace kllm::nn;
// No API changes required; internal graph ownership improved to prevent leaks.
```

Available ops and layers:
- Tensors: `tensor(values, shape, requires_grad)`, `zeros(shape)`, `ones(shape)`, `full(shape, value)`, `randu(shape)`, `randn(shape)`
- Ops: `add`, `mul`, `matmul`, `add_bias`, activations: `relu`, `leaky_relu`, `elu`, `selu`, `gelu`, softmax: `softmax_lastdim`; losses: `mse_loss`, `cross_entropy_logits`
- Modules: `Linear`, `Sequential`, `Dropout`, `LayerNorm`, `BatchNorm1d`; utilities: `collect_parameters`, `summary`
- Optimizers: `SGD(params, lr[, weight_decay])`, `Adam(params, lr)`, `RMSprop`, `Adagrad`, `Adadelta`
- Training helpers: gradient clipping: `clip_grad_norm(params, max_norm)`; LR schedulers: `StepLR`, `CosineAnnealingLR`
- Data: `TensorDataset::from`, `TensorDatasetReg::from` (regression), `DataLoader`, `DataLoaderReg`
- Trainer: `Trainer(cfg).fit(model, loader, optimizer)`, `fit_regression(model, loaderReg, optimizer, epochs)`

Notes:
- Each `nn::Value` exposes a `data()` method for raw access; internal storage is now `values`.
- `Module::parameters()` returns a vector; `collect_parameters(model)` gathers recursively.

Custom module implementation (example):
```cpp
struct MyReluModule : kllm::nn::Module {
  kllm::nn::ValuePtr forward(const kllm::nn::ValuePtr &x) override { return kllm::nn::relu(x); }
  std::vector<kllm::nn::ValuePtr> parameters() override { return {}; }
};
```

Autograd notes:
- Backprop builds a DAG of `Value` parents. Call `loss->backward()` to populate `.grad` for tensors with `requires_grad=true`.
- Gradients flow through ops via stored `backward_fn` lambdas; use `zero_grad()` on parameters before each step.
- For stability, you can use `clip_grad_norm(params, max_norm)` before optimizer `step()`.

This API is intentionally compact for easy use on mobile/Termux while staying header-only.

---

### Benchmarks (this build)
```text
FWHT 1M floats: ~2.64–5.22 ms
FWHT(par,4) 1M floats: ~2.64–3.66 ms
Fused FWHT-scale-add 1M: ~6.5 ms
NTT 262k uint32: ~4.5 ms
CountSketch 1M -> 262k: ~4.3 ms
BlockDiag float 1024x(16x16): ~0.049 ms
BlockDiag int8 1024x(16x16): ~0.046 ms
LowRank 4096x4096 (r=64): ~0.00003–0.00004 ms
Pipeline v2.1 int8 1M: ~24–30 ms (slabs=4)
Pipeline v2.1 int4 1M: ~23–28 ms (slabs=4)
```

Notes:
- `enable_simd=false` forces scalar paths for maximum portability/testing.
- Blocked matmul uses runtime tiling; adjust `matmul_block_*` to match cache.

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