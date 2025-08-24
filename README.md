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

Threading/config:
```cpp
kllm::set_num_threads(8);
kllm::set_parallel_threshold(1<<14);
kllm::set_large_slab_bytes(1024*1024);
kllm::set_enable_gpu(true); // if built with -DKLLM_USE_OPENCL
```

---

### nn/autograd quickstart
Build a tiny MLP classifier using the new minimal nn API:
```cpp
using namespace kllm::nn;

// Data: N samples of D features with integer labels in [0,C)
std::vector<std::vector<float>> X; std::vector<int> Y; /* fill */
auto ds = TensorDataset::from(X, Y);
DataLoader loader(ds, DataLoaderConfig{.batch=64, .shuffle=true});

// Model: D -> 64 -> C
size_t D = ds.d, C = 10;
auto net = Sequential({
  std::make_shared<Linear>(D, 64),
  // GELU via manual call
});

// Attach second layer
net.mods.push_back(std::make_shared<Linear>(64, C));

// Collect params and choose optimizer
auto params = collect_parameters(net);
Adam opt(params, 1e-3f);

// Train
Trainer::Config cfg; cfg.epochs = 5;
Trainer trainer(cfg);
auto metrics = trainer.fit(net, loader, opt);
std::cout << "Loss=" << metrics.loss / metrics.samples
          << ", Acc=" << metrics.acc << "\n";
```

Available ops and layers:
- Tensors: `tensor(values, shape, requires_grad)`, `zeros(shape)`, `randn(shape)`
- Ops: `add`, `mul`, `matmul`, `add_bias`, `relu`, `gelu`, `softmax_lastdim`, losses: `mse_loss`, `cross_entropy_logits`
- Modules: `Linear`, `Sequential`; utilities: `collect_parameters`, `summary`
- Optimizers: `SGD(params, lr[, weight_decay])`, `Adam(params, lr)`
- Data: `TensorDataset::from`, `DataLoader` with batch/shuffle
- Trainer: `Trainer(cfg).fit(model, loader, optimizer)`

This API is intentionally compact for easy use on mobile/Termux while staying header-only.

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