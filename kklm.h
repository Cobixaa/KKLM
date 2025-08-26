#pragma once

// Single-header KLLM runtime (kklm.h)
// Consolidated public API and implementations. Header-only.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <type_traits>
#include <thread>
#include <queue>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <random>
#include <ostream>
#include <iostream>
#include <sstream>

#if defined(__AVX2__)
	#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	#include <arm_neon.h>
#endif
#if defined(__linux__)
	#include <pthread.h>
	#include <sched.h>
#endif

namespace kllm {

// Strong inline hint for hot paths
#if defined(__GNUC__) || defined(__clang__)
	#define KLLM_INLINE inline __attribute__((always_inline))
#else
	#define KLLM_INLINE inline
#endif

// ===== utils =====
inline bool is_power_of_two(std::size_t value) {
	return value != 0 && (value & (value - 1)) == 0;
}

inline bool is_aligned_32(const void *ptr) {
	return (reinterpret_cast<std::uintptr_t>(ptr) & 31u) == 0u;
}

#if defined(__GNUC__) || defined(__clang__)
	#define KLLM_LIKELY(x) (__builtin_expect(!!(x), 1))
	#define KLLM_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
	#define KLLM_LIKELY(x) (x)
	#define KLLM_UNLIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
	inline void prefetch(const void *ptr, int locality = 3) {
		if (locality <= 0) {
			__builtin_prefetch(ptr, 0, 0);
		} else if (locality == 1) {
			__builtin_prefetch(ptr, 0, 1);
		} else if (locality == 2) {
			__builtin_prefetch(ptr, 0, 2);
		} else {
			__builtin_prefetch(ptr, 0, 3);
		}
	}
#else
	inline void prefetch(const void *ptr, int locality = 3) {
		(void)ptr; (void)locality;
	}
#endif

#if defined(__GNUC__) || defined(__clang__)
	#define KLLM_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#else
	#define KLLM_ASSUME_ALIGNED(ptr, align) (ptr)
#endif

#if defined(__GNUC__) || defined(__clang__)
	#define KLLM_RESTRICT __restrict__
#else
	#define KLLM_RESTRICT
#endif

// ===== status =====
enum class StatusCode {
	kOk = 0,
	kInvalidArgument = 1,
	kFailedPrecondition = 2,
	kInternal = 3
};

struct Status {
	StatusCode code;
	const char *message;

	constexpr bool ok() const { return code == StatusCode::kOk; }
	static constexpr Status OK() { return Status{StatusCode::kOk, "OK"}; }
};

inline Status make_status(StatusCode code, const char *msg) {
	return Status{code, msg};
}

// Thread-local small buffer for error messages that include errno or dynamic details.
// The returned c_str pointers are valid until the calling thread overwrites them with a new call.
inline const char * make_status_message(const char *base, const char *detail = nullptr, int errnum = 0) {
	static thread_local char buf[256];
	if (detail && errnum != 0) {
		std::snprintf(buf, sizeof(buf), "%s: %s (errno=%d)", base, detail, errnum);
	} else if (detail) {
		std::snprintf(buf, sizeof(buf), "%s: %s", base, detail);
	} else if (errnum != 0) {
		const char *emsg = std::strerror(errnum);
		std::snprintf(buf, sizeof(buf), "%s: %s (errno=%d)", base, emsg ? emsg : "?", errnum);
	} else {
		std::snprintf(buf, sizeof(buf), "%s", base);
	}
	return buf;
}

inline Status make_status_errno(StatusCode code, const char *base, int errnum) {
	return make_status(code, make_status_message(base, nullptr, errnum));
}

// Lightweight StatusOr to propagate values or errors without exceptions.
template <typename T>
class StatusOr {
public:
	StatusOr(const Status &s) : status_(s), has_value_(false) {}
	StatusOr(T &&value) : status_(Status::OK()), has_value_(true) {
		new (&storage_) T(std::move(value));
	}
	StatusOr(const T &value) : status_(Status::OK()), has_value_(true) {
		new (&storage_) T(value);
	}
	~StatusOr() { if (has_value_) { reinterpret_cast<T*>(&storage_)->~T(); } }

	bool ok() const { return status_.ok(); }
	const Status & status() const { return status_; }
	T & value() { return *reinterpret_cast<T*>(&storage_); }
	const T & value() const { return *reinterpret_cast<const T*>(&storage_); }

private:
	Status status_;
	bool has_value_;
	typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_;
};

#define KLLM_RETURN_IF_FALSE(cond, code, msg) \
	do { if (KLLM_UNLIKELY(!(cond))) { return make_status((code), (msg)); } } while(0)

#define KLLM_RETURN_IF_ERROR(expr) \
	do { Status _st = (expr); if (KLLM_UNLIKELY(!_st.ok())) return _st; } while(0)

// ===== config =====
struct Config {
	bool deterministic = false;
	std::size_t num_threads = 0; // 0 => use hardware_concurrency
	std::size_t prefetch_distance = 32; // elements ahead for prefetching (floats)
	std::size_t parallel_threshold = 1 << 14; // use parallel paths when length >= threshold
	bool pin_threads = false; // try to pin worker threads to cores on Linux
	// v2.1 additions
	bool prefer_hugepages = false; // hint for slab allocator (best-effort)
	std::size_t large_slab_bytes = 256 * 1024; // 256 KB default slab size
	std::size_t max_inflight_slabs = 3; // double/triple buffering
	bool enable_pipeline_nt_stores = true; // allow non-temporal stores at pipeline edges
	std::size_t sketch_num_hashes = 3; // CountSketch hashes for v2 path
	std::size_t routing_bucket_size = 256; // elements per routing bucket (approx L2 tile)
	// v2.2 additions (GPU removed; CPU-only)
	bool release_tls_fused_buffers = true; // release fused FWHT TLS buffers after each call
	bool enable_simd = true; // allow using SIMD paths when compiled in
	// matmul blocking knobs
	bool enable_matmul_blocked = true;
	std::size_t matmul_block_m = 64;
	std::size_t matmul_block_n = 128;
	std::size_t matmul_block_k = 128;
};

inline Config & global_config() {
	static Config cfg{};
	return cfg;
}

inline void set_deterministic(bool enabled) {
	global_config().deterministic = enabled;
}

inline void set_num_threads(std::size_t n) {
	global_config().num_threads = n;
}

inline void set_prefetch_distance(std::size_t elements_ahead) {
	global_config().prefetch_distance = elements_ahead;
}

inline void set_parallel_threshold(std::size_t threshold) {
	global_config().parallel_threshold = threshold;
}

inline void set_pin_threads(bool pin) {
	global_config().pin_threads = pin;
}

// v2.1 config setters
inline void set_large_slab_bytes(std::size_t bytes) { global_config().large_slab_bytes = bytes; }
inline void set_max_inflight_slabs(std::size_t n) { global_config().max_inflight_slabs = n == 0 ? 1 : n; }
inline void set_prefer_hugepages(bool enable) { global_config().prefer_hugepages = enable; }
inline void set_pipeline_nt_stores(bool enable) { global_config().enable_pipeline_nt_stores = enable; }
inline void set_sketch_num_hashes(std::size_t h) { global_config().sketch_num_hashes = (h < 1 ? 1 : (h > 4 ? 4 : h)); }
inline void set_routing_bucket_size(std::size_t b) { global_config().routing_bucket_size = (b == 0 ? 256 : b); }
// GPU setter removed; CPU-only

// ===== memory =====
struct FreeDeleter {
	void operator()(void *ptr) const noexcept {
		std::free(ptr);
	}
};

inline std::unique_ptr<void, FreeDeleter> allocate_aligned_bytes(std::size_t size_bytes, std::size_t alignment) {
	if (alignment < alignof(void *)) {
		alignment = alignof(void *);
	}
	void *ptr = nullptr;
#if defined(_POSIX_VERSION)
	if (posix_memalign(&ptr, alignment, size_bytes) != 0) {
		return std::unique_ptr<void, FreeDeleter>(nullptr);
	}
	return std::unique_ptr<void, FreeDeleter>(ptr);
#else
	// Fallback to standard aligned_alloc if available (requires size to be multiple of alignment)
	std::size_t rounded = (size_bytes + alignment - 1u) / alignment * alignment;
	ptr = std::aligned_alloc(alignment, rounded);
	return std::unique_ptr<void, FreeDeleter>(ptr);
#endif
}

inline StatusOr<std::unique_ptr<void, FreeDeleter>> allocate_aligned_bytes_status(std::size_t size_bytes, std::size_t alignment) {
	if (alignment < alignof(void *)) alignment = alignof(void *);
	if (size_bytes == 0) {
		return StatusOr<std::unique_ptr<void, FreeDeleter>>(make_status(StatusCode::kInvalidArgument, "allocate_aligned_bytes: size_bytes must be > 0"));
	}
	void *ptr = nullptr;
#if defined(_POSIX_VERSION)
	int rc = posix_memalign(&ptr, alignment, size_bytes);
	if (rc != 0) {
		return StatusOr<std::unique_ptr<void, FreeDeleter>>(make_status_errno(StatusCode::kFailedPrecondition, "posix_memalign failed", rc));
	}
	return std::unique_ptr<void, FreeDeleter>(ptr);
#else
	std::size_t rounded = (size_bytes + alignment - 1u) / alignment * alignment;
	ptr = std::aligned_alloc(alignment, rounded);
	if (!ptr) {
		return StatusOr<std::unique_ptr<void, FreeDeleter>>(make_status(StatusCode::kFailedPrecondition, "aligned_alloc failed"));
	}
	return std::unique_ptr<void, FreeDeleter>(ptr);
#endif
}

template <typename T>
inline std::unique_ptr<T, FreeDeleter> allocate_aligned(std::size_t count, std::size_t alignment) {
	const std::size_t total = count * sizeof(T);
	auto raw = allocate_aligned_bytes(total, alignment);
	return std::unique_ptr<T, FreeDeleter>(static_cast<T *>(raw.release()));
}

inline Status set_current_thread_affinity_status(int cpu_index) {
	(void)cpu_index;
#if defined(__linux__)
	if (cpu_index < 0) {
		return make_status(StatusCode::kInvalidArgument, "cpu_index must be >= 0");
	}
#if defined(CPU_SETSIZE)
	if (static_cast<unsigned int>(cpu_index) >= CPU_SETSIZE) {
		return make_status(StatusCode::kInvalidArgument, "cpu_index exceeds CPU_SETSIZE");
	}
#endif
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(static_cast<unsigned int>(cpu_index), &set);
	const int result = sched_setaffinity(0, sizeof(cpu_set_t), &set);
	if (result != 0) {
		return make_status_errno(StatusCode::kFailedPrecondition, "sched_setaffinity failed", errno);
	}
	return Status::OK();
#else
	return make_status(StatusCode::kFailedPrecondition, "Thread affinity not supported on this platform");
#endif
}

inline bool set_current_thread_affinity(int cpu_index) {
	return set_current_thread_affinity_status(cpu_index).ok();
}

// ===== parallel =====
class ThreadPool {
public:
	explicit ThreadPool(std::size_t num_threads) : stop_flag(false), outstanding_tasks(0)
		, pin_workers(global_config().pin_threads) {
		if (num_threads == 0) num_threads = 1;
		workers.reserve(num_threads);
		for (std::size_t i = 0; i < num_threads; ++i) {
			workers.emplace_back([this, i]() {
				if (pin_workers) {
					(void)set_current_thread_affinity(static_cast<int>(i));
				}
				this->worker_loop();
			});
		}
	}

	~ThreadPool() {
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			stop_flag = true;
		}
		cv.notify_all();
		for (auto &t : workers) {
			if (t.joinable()) t.join();
		}
	}

	void enqueue(std::function<void()> fn) {
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			++outstanding_tasks;
			tasks.push(std::move(fn));
		}
		cv.notify_one();
	}

	void wait() {
		std::unique_lock<std::mutex> lock(done_mutex);
		done_cv.wait(lock, [this]() { return outstanding_tasks.load() == 0; });
	}

	bool wait_for(std::chrono::milliseconds timeout) {
		std::unique_lock<std::mutex> lock(done_mutex);
		return done_cv.wait_for(lock, timeout, [this]() { return outstanding_tasks.load() == 0; });
	}

	std::size_t size() const { return workers.size(); }

private:
	void worker_loop() {
		for (;;) {
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				cv.wait(lock, [this]() { return stop_flag || !tasks.empty(); });
				if (stop_flag && tasks.empty()) return;
				task = std::move(tasks.front());
				tasks.pop();
			}
			try {
				task();
			} catch (...) {
				// Swallow exceptions to keep thread alive; mark task done.
			}
			if (outstanding_tasks.fetch_sub(1) == 1) {
				std::lock_guard<std::mutex> g(done_mutex);
				done_cv.notify_all();
			}
		}
	}

	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;
	mutable std::mutex queue_mutex;
	std::condition_variable cv;
	std::atomic<bool> stop_flag;
	std::atomic<std::size_t> outstanding_tasks;
	std::condition_variable done_cv;
	std::mutex done_mutex;
	bool pin_workers;
};

inline void parallel_for_blocks(ThreadPool &pool, std::size_t begin, std::size_t end, std::size_t step, std::function<void(std::size_t)> fn) {
	if (begin >= end || step == 0) return;
	// Chunk by number of workers
	const std::size_t num_workers = pool.size();
	const std::size_t num_iters = (end - begin + step - 1) / step;
	const std::size_t chunk_iters = (num_iters + num_workers - 1) / num_workers;
	std::size_t start_iter = 0;
	while (start_iter < num_iters) {
		const std::size_t this_iters = (chunk_iters < (num_iters - start_iter)) ? chunk_iters : (num_iters - start_iter);
		const std::size_t base = begin + start_iter * step;
		pool.enqueue([=]() {
			for (std::size_t j = 0; j < this_iters; ++j) {
				fn(base + j * step);
			}
		});
		start_iter += this_iters;
	}
	pool.wait();
}

// Thread-local pool reuse to avoid repeated construction costs on hot fused paths.
inline ThreadPool & get_thread_local_pool(std::size_t requested_threads) {
	thread_local std::unique_ptr<ThreadPool> pool;
	thread_local std::size_t configured_threads = 0;
	if (!pool || configured_threads != requested_threads) {
		pool.reset(new ThreadPool(requested_threads));
		configured_threads = requested_threads;
	}
	return *pool;
}

// ===== quant =====
struct QuantParams {
	float scale;
};

inline QuantParams choose_symmetric_int8_scale(const float *data, std::size_t length) {
	float max_abs = 0.0f;
	for (std::size_t i = 0; i < length; ++i) {
		const float v = data[i] < 0.0f ? -data[i] : data[i];
		if (v > max_abs) max_abs = v;
	}
	QuantParams p{};
	float s = (max_abs <= 1e-8f) ? 1.0f : (max_abs / 127.0f);
	if (!std::isfinite(s) || s <= 0.0f) s = 1.0f;
	p.scale = s;
	return p;
}

inline void quantize_int8(const float *input, std::size_t length, int8_t *output, const QuantParams &params) {
	if (input == nullptr || output == nullptr) return;
	const float inv_scale = params.scale == 0.0f ? 0.0f : (1.0f / params.scale);
	std::size_t i = 0;
#if defined(__AVX2__)
	const __m256 v_inv = _mm256_set1_ps(inv_scale);
	for (; i + 8 <= length; i += 8) {
		__m256 x = _mm256_loadu_ps(input + i);
		__m256 y = _mm256_mul_ps(x, v_inv);
		__m256 y_round = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
		__m256i y_i32 = _mm256_cvtps_epi32(y_round);
		__m256i y_clamped = _mm256_min_epi32(_mm256_max_epi32(y_i32, _mm256_set1_epi32(-127)), _mm256_set1_epi32(127));
		alignas(32) int32_t tmp[8];
		_mm256_store_si256(reinterpret_cast<__m256i *>(tmp), y_clamped);
		for (int k = 0; k < 8; ++k) {
			output[i + static_cast<std::size_t>(k)] = static_cast<int8_t>(tmp[k]);
		}
	}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
	for (; i + 8 <= length; i += 8) {
		float32x4_t x0 = vld1q_f32(input + i);
		float32x4_t x1 = vld1q_f32(input + i + 4);
		float32x4_t invv = vdupq_n_f32(inv_scale);
		float32x4_t y0f = vmulq_f32(x0, invv);
		float32x4_t y1f = vmulq_f32(x1, invv);
#if defined(__aarch64__)
		int32x4_t y0 = vcvtnq_s32_f32(y0f);
		int32x4_t y1 = vcvtnq_s32_f32(y1f);
#else
		int32x4_t y0 = vcvtq_s32_f32(vrndnq_f32(y0f));
		int32x4_t y1 = vcvtq_s32_f32(vrndnq_f32(y1f));
#endif
		int32x4_t minv = vdupq_n_s32(-127);
		int32x4_t maxv = vdupq_n_s32(127);
		y0 = vmaxq_s32(minv, vminq_s32(y0, maxv));
		y1 = vmaxq_s32(minv, vminq_s32(y1, maxv));
		int16x4_t y0_16 = vqmovn_s32(y0);
		int16x4_t y1_16 = vqmovn_s32(y1);
		int16x8_t y_16x8 = vcombine_s16(y0_16, y1_16);
		int8x8_t y_8x8 = vqmovn_s16(y_16x8);
		vst1_s8(reinterpret_cast<int8_t *>(output + i), y_8x8);
	}
#endif
	for (; i < length; ++i) {
		int v = static_cast<int>(input[i] * inv_scale + (input[i] >= 0.0f ? 0.5f : -0.5f));
		if (v > 127) v = 127;
		if (v < -127) v = -127;
		output[i] = static_cast<int8_t>(v);
	}
}

inline void dequantize_int8(const int8_t *input, std::size_t length, float scale, float *output) {
	if (input == nullptr || output == nullptr) return;
	float s = (!std::isfinite(scale) || scale <= 0.0f) ? 1.0f : scale;
	for (std::size_t j = 0; j < length; ++j) {
		output[j] = static_cast<float>(input[j]) * s;
	}
}

// ===== fast_transform (FWHT) =====
KLLM_INLINE void fwht_inplace_parallel(float * KLLM_RESTRICT data, std::size_t length, ThreadPool &pool);
KLLM_INLINE void fwht_inplace(float * KLLM_RESTRICT data, std::size_t length) {
	if (data == nullptr || !is_power_of_two(length)) {
		return;
	}
	// Auto-parallelize for large inputs
	if (length >= global_config().parallel_threshold) {
		std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		ThreadPool &pool = get_thread_local_pool(threads);
		fwht_inplace_parallel(data, length, pool);
		return;
	}
	for (std::size_t half_block = 1; half_block < length; half_block <<= 1) {
		const std::size_t full_block = half_block << 1;
		for (std::size_t block_start = 0; block_start < length; block_start += full_block) {
			std::size_t i = 0;
#if defined(__AVX2__)
			// Vectorize the butterfly with AVX where possible (8 floats per vector)
			const std::size_t vec_width = 8;
			for (; i + vec_width <= half_block; i += vec_width) {
				float *a_ptr = data + block_start + i;
				float *b_ptr = data + block_start + i + half_block;
				prefetch(a_ptr + global_config().prefetch_distance);
				prefetch(b_ptr + global_config().prefetch_distance);
				const bool aligned = is_aligned_32(a_ptr) && is_aligned_32(b_ptr);
				__m256 a = aligned ? _mm256_load_ps(a_ptr) : _mm256_loadu_ps(a_ptr);
				__m256 b = aligned ? _mm256_load_ps(b_ptr) : _mm256_loadu_ps(b_ptr);
				__m256 sum = _mm256_add_ps(a, b);
				__m256 diff = _mm256_sub_ps(a, b);
				if (aligned) {
					_mm256_store_ps(a_ptr, sum);
					_mm256_store_ps(b_ptr, diff);
				} else {
					_mm256_storeu_ps(a_ptr, sum);
					_mm256_storeu_ps(b_ptr, diff);
				}
			}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
			// NEON path (4 floats per vector)
			const std::size_t neon_width = 4;
			for (; i + neon_width <= half_block; i += neon_width) {
				float *a_ptr = data + block_start + i;
				float *b_ptr = data + block_start + i + half_block;
				prefetch(a_ptr + global_config().prefetch_distance);
				prefetch(b_ptr + global_config().prefetch_distance);
				float32x4_t a = vld1q_f32(a_ptr);
				float32x4_t b = vld1q_f32(b_ptr);
				float32x4_t sum = vaddq_f32(a, b);
				float32x4_t diff = vsubq_f32(a, b);
				vst1q_f32(a_ptr, sum);
				vst1q_f32(b_ptr, diff);
			}
#endif
			for (; i < half_block; ++i) {
				float &a = data[block_start + i];
				float &b = data[block_start + i + half_block];
				const float sum = a + b;
				const float diff = a - b;
				a = sum;
				b = diff;
			}
		}
	}
}

KLLM_INLINE void fwht_inplace_parallel(float * KLLM_RESTRICT data, std::size_t length, ThreadPool &pool) {
	if (data == nullptr || !is_power_of_two(length)) {
		return;
	}
	for (std::size_t half_block = 1; half_block < length; half_block <<= 1) {
		const std::size_t full_block = half_block << 1;
		parallel_for_blocks(pool, 0, length, full_block, [=](std::size_t block_start) {
			std::size_t i = 0;
#if defined(__AVX2__)
			if (global_config().enable_simd) {
				const std::size_t vec_width = 8;
				for (; i + vec_width <= half_block; i += vec_width) {
					float *a_ptr = data + block_start + i;
					float *b_ptr = data + block_start + i + half_block;
					prefetch(a_ptr + global_config().prefetch_distance);
					prefetch(b_ptr + global_config().prefetch_distance);
					const bool aligned = is_aligned_32(a_ptr) && is_aligned_32(b_ptr);
					__m256 a = aligned ? _mm256_load_ps(a_ptr) : _mm256_loadu_ps(a_ptr);
					__m256 b = aligned ? _mm256_load_ps(b_ptr) : _mm256_loadu_ps(b_ptr);
					__m256 sum = _mm256_add_ps(a, b);
					__m256 diff = _mm256_sub_ps(a, b);
					if (aligned) { _mm256_store_ps(a_ptr, sum); _mm256_store_ps(b_ptr, diff); }
					else { _mm256_storeu_ps(a_ptr, sum); _mm256_storeu_ps(b_ptr, diff); }
				}
			}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
			if (global_config().enable_simd) {
				const std::size_t neon_width = 4;
				for (; i + neon_width <= half_block; i += neon_width) {
					float *a_ptr = data + block_start + i;
					float *b_ptr = data + block_start + i + half_block;
					prefetch(a_ptr + global_config().prefetch_distance);
					prefetch(b_ptr + global_config().prefetch_distance);
					float32x4_t a = vld1q_f32(a_ptr);
					float32x4_t b = vld1q_f32(b_ptr);
					float32x4_t sum = vaddq_f32(a, b);
					float32x4_t diff = vsubq_f32(a, b);
					vst1q_f32(a_ptr, sum);
					vst1q_f32(b_ptr, diff);
				}
			}
#endif
			for (; i < half_block; ++i) {
				float &a = data[block_start + i];
				float &b = data[block_start + i + half_block];
				const float sum = a + b;
				const float diff = a - b;
				a = sum;
				b = diff;
			}
		});
	}
}

KLLM_INLINE void fwht_inplace_inverse(float * KLLM_RESTRICT data, std::size_t length) {
	if (data == nullptr || !is_power_of_two(length)) {
		return;
	}
	fwht_inplace(data, length);
	const float scale = 1.0f / static_cast<float>(length);
	for (std::size_t i = 0; i < length; ++i) {
		data[i] *= scale;
	}
}

// ===== fused kernels =====
KLLM_INLINE void fused_fwht_scale_add(const float * KLLM_RESTRICT input, std::size_t length, float scale, float * KLLM_RESTRICT inout_destination) {
	if (input == nullptr || inout_destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	static thread_local std::vector<float> buffer;
	buffer.resize(length);
	std::memcpy(buffer.data(), input, length * sizeof(float));
	const bool use_parallel = (length >= global_config().parallel_threshold);
	if (use_parallel) {
		std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		ThreadPool &pool = get_thread_local_pool(threads);
		fwht_inplace_parallel(buffer.data(), length, pool);
	} else {
		fwht_inplace(buffer.data(), length);
	}
	std::size_t i = 0;
#if defined(__AVX2__)
	const __m256 vscale = _mm256_set1_ps(scale);
	for (; i + 8 <= length; i += 8) {
		prefetch(buffer.data() + i + global_config().prefetch_distance);
		prefetch(inout_destination + i + global_config().prefetch_distance);
		__m256 u = _mm256_loadu_ps(buffer.data() + i);
		__m256 acc = _mm256_loadu_ps(inout_destination + i);
		acc = _mm256_fmadd_ps(u, vscale, acc);
		_mm256_storeu_ps(inout_destination + i, acc);
	}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
	float32x4_t vscale = vdupq_n_f32(scale);
	for (; i + 4 <= length; i += 4) {
		prefetch(buffer.data() + i + global_config().prefetch_distance);
		prefetch(inout_destination + i + global_config().prefetch_distance);
		float32x4_t u = vld1q_f32(buffer.data() + i);
		float32x4_t acc = vld1q_f32(inout_destination + i);
		acc = vmlaq_f32(acc, u, vscale);
		vst1q_f32(inout_destination + i, acc);
	}
#endif
	for (; i < length; ++i) {
		inout_destination[i] += scale * buffer[i];
	}
	if (global_config().release_tls_fused_buffers) {
		std::vector<float>().swap(buffer);
	}
}

KLLM_INLINE void fused_fwht_bias_relu(const float * KLLM_RESTRICT input, const float * KLLM_RESTRICT bias, std::size_t length, float * KLLM_RESTRICT destination) {
	if (input == nullptr || bias == nullptr || destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	static thread_local std::vector<float> buffer;
	buffer.resize(length);
	std::memcpy(buffer.data(), input, length * sizeof(float));
	if (length >= global_config().parallel_threshold) {
		std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		ThreadPool &pool = get_thread_local_pool(threads);
		fwht_inplace_parallel(buffer.data(), length, pool);
	} else {
		fwht_inplace(buffer.data(), length);
	}
	std::size_t i = 0;
#if defined(__AVX2__)
	for (; i + 8 <= length; i += 8) {
		prefetch(buffer.data() + i + global_config().prefetch_distance);
		prefetch(bias + i + global_config().prefetch_distance);
		__m256 u = _mm256_loadu_ps(buffer.data() + i);
		__m256 b = _mm256_loadu_ps(bias + i);
		__m256 y = _mm256_add_ps(u, b);
		__m256 zero = _mm256_setzero_ps();
		y = _mm256_max_ps(y, zero);
		_mm256_storeu_ps(destination + i, y);
	}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
	for (; i + 4 <= length; i += 4) {
		prefetch(buffer.data() + i + global_config().prefetch_distance);
		prefetch(bias + i + global_config().prefetch_distance);
		float32x4_t u = vld1q_f32(buffer.data() + i);
		float32x4_t b = vld1q_f32(bias + i);
		float32x4_t y = vaddq_f32(u, b);
		float32x4_t zero = vdupq_n_f32(0.0f);
		y = vmaxq_f32(y, zero);
		vst1q_f32(destination + i, y);
	}
#endif
	for (; i < length; ++i) {
		float y = buffer[i] + bias[i];
		destination[i] = y < 0.0f ? 0.0f : y;
	}
	if (global_config().release_tls_fused_buffers) {
		std::vector<float>().swap(buffer);
	}
}

KLLM_INLINE void fused_fwht_bias_gelu(const float * KLLM_RESTRICT input, const float * KLLM_RESTRICT bias, std::size_t length, float * KLLM_RESTRICT destination) {
	if (input == nullptr || bias == nullptr || destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	static thread_local std::vector<float> buffer;
	buffer.resize(length);
	std::memcpy(buffer.data(), input, length * sizeof(float));
	if (length >= global_config().parallel_threshold) {
		std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		ThreadPool &pool = get_thread_local_pool(threads);
		fwht_inplace_parallel(buffer.data(), length, pool);
	} else {
		fwht_inplace(buffer.data(), length);
	}
	for (std::size_t i = 0; i < length; ++i) {
		float x = buffer[i] + bias[i];
		float t = x * (0.7978845608028654f) * (1.0f + 0.044715f * x * x); // sqrt(2/pi)
		float y = 0.5f * x * (1.0f + std::tanh(t));
		destination[i] = y;
	}
	if (global_config().release_tls_fused_buffers) {
		std::vector<float>().swap(buffer);
	}
}

KLLM_INLINE void fused_fwht_bias_silu(const float * KLLM_RESTRICT input, const float * KLLM_RESTRICT bias, std::size_t length, float * KLLM_RESTRICT destination) {
	if (input == nullptr || bias == nullptr || destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	static thread_local std::vector<float> buffer;
	buffer.resize(length);
	std::memcpy(buffer.data(), input, length * sizeof(float));
	if (length >= global_config().parallel_threshold) {
		std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		ThreadPool &pool = get_thread_local_pool(threads);
		fwht_inplace_parallel(buffer.data(), length, pool);
	} else {
		fwht_inplace(buffer.data(), length);
	}
	for (std::size_t i = 0; i < length; ++i) {
		float x = buffer[i] + bias[i];
		float s = 1.0f / (1.0f + std::exp(-x));
		destination[i] = x * s; // swish
	}
	if (global_config().release_tls_fused_buffers) {
		std::vector<float>().swap(buffer);
	}
}

KLLM_INLINE float l2_norm(const float * KLLM_RESTRICT x, std::size_t length) {
	if (x == nullptr) return 0.0f;
	double acc = 0.0;
	for (std::size_t i = 0; i < length; ++i) {
		double v = static_cast<double>(x[i]);
		acc += v * v;
	}
	return static_cast<float>(std::sqrt(acc));
}

// ===== NTT =====
static constexpr std::uint32_t kMod = 998244353u; // 119 * 2^23 + 1, primitive root g = 3
static constexpr std::uint32_t kPrimitiveRoot = 3u;

inline std::uint32_t add_mod(std::uint32_t a, std::uint32_t b) {
	std::uint32_t c = a + b;
	if (c >= kMod) c -= kMod;
	return c;
}

inline std::uint32_t sub_mod(std::uint32_t a, std::uint32_t b) {
	return (a >= b) ? (a - b) : (a + kMod - b);
}

inline std::uint32_t mul_mod(std::uint64_t a, std::uint64_t b) {
	return static_cast<std::uint32_t>((a * b) % kMod);
}

inline std::uint32_t pow_mod(std::uint32_t base, std::uint32_t exp) {
	std::uint64_t result = 1u;
	std::uint64_t cur = base;
	while (exp > 0) {
		if (exp & 1u) {
			result = (result * cur) % kMod;
		}
		cur = (cur * cur) % kMod;
		exp >>= 1u;
	}
	return static_cast<std::uint32_t>(result);
}

inline std::uint32_t inv_mod(std::uint32_t x) {
	// Fermat inverse: x^(mod-2)
	return pow_mod(x, kMod - 2u);
}

inline void bit_reverse_permute(std::vector<std::uint32_t> &a) {
	const std::size_t n = a.size();
	std::size_t j = 0;
	for (std::size_t i = 1; i < n; ++i) {
		std::size_t bit = n >> 1;
		for (; j & bit; bit >>= 1) {
			j ^= bit;
		}
		j ^= bit;
		if (i < j) {
			std::uint32_t tmp = a[i];
			a[i] = a[j];
			a[j] = tmp;
		}
	}
}

inline bool ntt_inplace(std::vector<std::uint32_t> &a, bool invert) {
	const std::size_t n = a.size();
	if (!is_power_of_two(n)) {
		return false;
	}
	bit_reverse_permute(a);

	for (std::size_t len = 2; len <= n; len <<= 1) {
		const std::size_t half = len >> 1;
		const std::uint32_t wlen = invert
			? inv_mod(pow_mod(kPrimitiveRoot, (kMod - 1u) / static_cast<std::uint32_t>(len)))
			: pow_mod(kPrimitiveRoot, (kMod - 1u) / static_cast<std::uint32_t>(len));
		// Precompute stage twiddles once and reuse across blocks
		std::vector<std::uint32_t> twiddles(half);
		twiddles[0] = 1u;
		for (std::size_t j = 1; j < half; ++j) {
			twiddles[j] = mul_mod(twiddles[j - 1], wlen);
		}
		for (std::size_t i = 0; i < n; i += len) {
			for (std::size_t j = 0; j < half; ++j) {
				const std::uint32_t u = a[i + j];
				const std::uint32_t v = mul_mod(a[i + j + half], twiddles[j]);
				a[i + j] = add_mod(u, v);
				a[i + j + half] = sub_mod(u, v);
			}
		}
	}
	if (invert) {
		const std::uint32_t inv_n = inv_mod(static_cast<std::uint32_t>(n % kMod));
		for (std::size_t i = 0; i < n; ++i) {
			a[i] = mul_mod(a[i], inv_n);
		}
	}
	return true;
}

// ===== sketch =====
inline std::uint64_t splitmix64(std::uint64_t x) {
	x += 0x9e3779b97f4a7c15ull;
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
	x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
	x = x ^ (x >> 31);
	return x;
}

struct CountSketch {
	std::size_t sketch_size;
	std::size_t num_hashes;
	std::vector<std::uint64_t> seeds;

	explicit CountSketch(std::size_t sketch_size_, std::size_t num_hashes_, std::uint64_t seed_base = 0x12345678abcdef00ull)
		: sketch_size(sketch_size_) , num_hashes(num_hashes_), seeds(num_hashes_, 0) {
		for (std::size_t i = 0; i < num_hashes; ++i) {
			seeds[i] = splitmix64(seed_base + static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull);
		}
	}

	inline void apply(const float *input, std::size_t length, float *output) const {
		if (input == nullptr || output == nullptr || sketch_size == 0 || num_hashes == 0) {
			return;
		}
		for (std::size_t i = 0; i < sketch_size; ++i) {
			output[i] = 0.0f;
		}
		for (std::size_t i = 0; i < length; ++i) {
			for (std::size_t h = 0; h < num_hashes; ++h) {
				const std::uint64_t mix = splitmix64(static_cast<std::uint64_t>(i) ^ seeds[h]);
				const std::size_t bucket = static_cast<std::size_t>(mix % static_cast<std::uint64_t>(sketch_size));
				const float sign = ((mix >> 63) == 0ull) ? 1.0f : -1.0f;
				output[bucket] += sign * input[i];
			}
		}
	}
};

// ===== transform_extras =====
KLLM_INLINE void block_diagonal_matvec(const float * KLLM_RESTRICT blocks_data, const std::size_t * KLLM_RESTRICT offsets, const std::size_t * KLLM_RESTRICT block_sizes, std::size_t num_blocks,
	const float * KLLM_RESTRICT x, float * KLLM_RESTRICT y) {
	std::size_t x_offset = 0;
	std::size_t y_offset = 0;
	for (std::size_t b = 0; b < num_blocks; ++b) {
		const std::size_t n = block_sizes[b];
		const float *block = blocks_data + offsets[b];
		for (std::size_t i = 0; i < n; ++i) {
			float acc = 0.0f;
			const float *row = block + i * n;
			for (std::size_t j = 0; j < n; ++j) {
				prefetch(row + j + 32);
				prefetch(x + x_offset + j + 32);
				acc += row[j] * x[x_offset + j];
			}
			y[y_offset + i] = acc;
		}
		x_offset += n;
		y_offset += n;
	}
}

// Mixed-precision: int8 weights with per-block scale, accumulate in float for quality
KLLM_INLINE void block_diagonal_matvec_int8(const int8_t * KLLM_RESTRICT blocks_q, const float * KLLM_RESTRICT scales, const std::size_t * KLLM_RESTRICT offsets, const std::size_t * KLLM_RESTRICT block_sizes, std::size_t num_blocks,
	const float * KLLM_RESTRICT x, float * KLLM_RESTRICT y) {
	std::size_t x_offset = 0;
	std::size_t y_offset = 0;
	for (std::size_t b = 0; b < num_blocks; ++b) {
		const std::size_t n = block_sizes[b];
		const float s = scales[b];
		const int8_t *block = blocks_q + offsets[b];
		for (std::size_t i = 0; i < n; ++i) {
			float acc = 0.0f;
			const int8_t *row = block + i * n;
			for (std::size_t j = 0; j < n; ++j) {
				prefetch(row + j + 64);
				prefetch(x + x_offset + j + 64);
				acc += static_cast<float>(row[j]) * s * x[x_offset + j];
			}
			y[y_offset + i] = acc;
		}
		x_offset += n;
		y_offset += n;
	}
}

KLLM_INLINE void low_rank_apply(const float * KLLM_RESTRICT U, const float * KLLM_RESTRICT V, std::size_t out_dim, std::size_t in_dim, std::size_t rank,
	const float * KLLM_RESTRICT x, float * KLLM_RESTRICT y) {
	// t = V^T x => length rank
	std::vector<float> t(rank, 0.0f);
	for (std::size_t r = 0; r < rank; ++r) {
		float acc = 0.0f;
		for (std::size_t j = 0; j < in_dim; ++j) {
			acc += V[j * rank + r] * x[j];
		}
		t[r] = acc;
	}
	for (std::size_t i = 0; i < out_dim; ++i) {
		float acc = 0.0f;
		const float *u_row = U + i * rank;
		for (std::size_t r = 0; r < rank; ++r) {
			acc += u_row[r] * t[r];
		}
		y[i] = acc;
	}
}

// ===== IR =====
struct Tensor {
	std::vector<float> values;

	Tensor() = default;
	explicit Tensor(std::size_t n) : values(n, 0.0f) {}
	explicit Tensor(std::vector<float> v) : values(std::move(v)) {}

	std::size_t size() const { return values.size(); }
	float * data() { return values.data(); }
	const float * data() const { return values.data(); }
};

struct Node {
	virtual ~Node() = default;
	virtual Tensor evaluate() = 0;
};

struct InputNode : public Node {
	Tensor tensor;
	explicit InputNode(Tensor t) : tensor(std::move(t)) {}

	Tensor evaluate() override {
		return tensor;
	}
};

struct TransformNode : public Node {
	std::shared_ptr<Node> input;
	explicit TransformNode(std::shared_ptr<Node> in) : input(std::move(in)) {}

	Tensor evaluate() override {
		Tensor x = input->evaluate();
		fwht_inplace(x.data(), x.size());
		return x;
	}
};

struct ReluNode : public Node {
	std::shared_ptr<Node> input;
	explicit ReluNode(std::shared_ptr<Node> in) : input(std::move(in)) {}

	Tensor evaluate() override {
		Tensor x = input->evaluate();
		for (float &v : x.values) {
			if (v < 0.0f) v = 0.0f;
		}
		return x;
	}
};

struct FusedTransformReluNode : public Node {
	std::shared_ptr<Node> input;
	explicit FusedTransformReluNode(std::shared_ptr<Node> in) : input(std::move(in)) {}

	Tensor evaluate() override {
		Tensor x = input->evaluate();
		fwht_inplace(x.data(), x.size());
		for (float &v : x.values) {
			if (v < 0.0f) v = 0.0f;
		}
		return x;
	}
};

struct GraphBuilder {
	static std::shared_ptr<Node> input(const std::vector<float> &values) {
		return std::make_shared<InputNode>(Tensor(values));
	}
	static std::shared_ptr<Node> transform(const std::shared_ptr<Node> &in) {
		return std::make_shared<TransformNode>(in);
	}
	static std::shared_ptr<Node> relu(const std::shared_ptr<Node> &in) {
		return std::make_shared<ReluNode>(in);
	}
};

struct Planner {
	// If pattern is Transform(Relu(x)) or Relu(Transform(x)), use fused node.
	static std::shared_ptr<Node> plan(const std::shared_ptr<Node> &root) {
		// Only recognize ReluNode over TransformNode and vice versa for this demo.
		auto try_cast_relu = std::dynamic_pointer_cast<ReluNode>(root);
		if (try_cast_relu) {
			auto try_cast_transform = std::dynamic_pointer_cast<TransformNode>(try_cast_relu->input);
			if (try_cast_transform) {
				return std::make_shared<FusedTransformReluNode>(try_cast_transform->input);
			}
			return root;
		}
		auto try_cast_transform = std::dynamic_pointer_cast<TransformNode>(root);
		if (try_cast_transform) {
			auto try_cast_relu2 = std::dynamic_pointer_cast<ReluNode>(try_cast_transform->input);
			if (try_cast_relu2) {
				return std::make_shared<FusedTransformReluNode>(try_cast_relu2->input);
			}
			return root;
		}
		return root;
	}
};

// ===== profiler =====
struct KernelTiming {
	double ms;
	std::size_t bytes;
};

class Profiler {
public:
	void record(const std::string &name, double ms, std::size_t bytes = 0) {
		db[name] = KernelTiming{ms, bytes};
	}

	KernelTiming get(const std::string &name) const {
		auto it = db.find(name);
		if (it == db.end()) return KernelTiming{0.0, 0};
		return it->second;
	}

private:
	std::unordered_map<std::string, KernelTiming> db;
};

struct ScopeTimer {
	std::chrono::steady_clock::time_point start;
	Profiler *prof;
	std::string name;
	std::size_t bytes;

	ScopeTimer(Profiler &p, std::string n, std::size_t b = 0)
		: start(std::chrono::steady_clock::now()), prof(&p), name(std::move(n)), bytes(b) {}
	~ScopeTimer() {
		auto end = std::chrono::steady_clock::now();
		double ms = std::chrono::duration<double, std::milli>(end - start).count();
		if (prof) prof->record(name, ms, bytes);
	}
};

// ===== scheduler =====
struct ScheduledPlan {
	std::shared_ptr<Node> root;
	bool used_fusion = false;
};

class Scheduler {
public:
	explicit Scheduler(Profiler &p) : prof(p) {}

	ScheduledPlan plan(const std::shared_ptr<Node> &root) {
		ScheduledPlan sp;
		// For demo: if input tensor size is power-of-two and large, favor FWHT based fused path via existing Planner
		auto relu = std::dynamic_pointer_cast<ReluNode>(root);
		if (relu) {
			auto tr = std::dynamic_pointer_cast<TransformNode>(relu->input);
			if (tr) {
				sp.root = std::make_shared<FusedTransformReluNode>(tr->input);
				sp.used_fusion = true;
				return sp;
			}
		}
		sp.root = root;
		sp.used_fusion = false;
		return sp;
	}

private:
#if defined(__GNUC__) || defined(__clang__)
	[[maybe_unused]]
#endif
	Profiler &prof;
};

// ===== api =====
inline Status fwht(std::vector<float> &inout) {
	if (!is_power_of_two(inout.size())) {
		return make_status(StatusCode::kInvalidArgument, "FWHT length must be power of two");
	}
	fwht_inplace(inout.data(), inout.size());
	return Status::OK();
}

inline Status fwht_inverse(std::vector<float> &inout) {
	if (!is_power_of_two(inout.size())) {
		return make_status(StatusCode::kInvalidArgument, "FWHT length must be power of two");
	}
	fwht_inplace_inverse(inout.data(), inout.size());
	return Status::OK();
}

inline Status fused_transform_bias_relu(const std::vector<float> &input, const std::vector<float> &bias, std::vector<float> &out) {
	if (input.size() != bias.size()) {
		return make_status(StatusCode::kInvalidArgument, "Input and bias must have same size");
	}
	if (!is_power_of_two(input.size())) {
		return make_status(StatusCode::kInvalidArgument, "Input length must be power of two");
	}
	out.resize(input.size());
	fused_fwht_bias_relu(input.data(), bias.data(), input.size(), out.data());
	return Status::OK();
}

inline Status fused_transform_bias_gelu(const std::vector<float> &input, const std::vector<float> &bias, std::vector<float> &out) {
	if (input.size() != bias.size()) {
		return make_status(StatusCode::kInvalidArgument, "Input and bias must have same size");
	}
	if (!is_power_of_two(input.size())) {
		return make_status(StatusCode::kInvalidArgument, "Input length must be power of two");
	}
	out.resize(input.size());
	fused_fwht_bias_gelu(input.data(), bias.data(), input.size(), out.data());
	return Status::OK();
}

inline Status fused_transform_bias_silu(const std::vector<float> &input, const std::vector<float> &bias, std::vector<float> &out) {
	if (input.size() != bias.size()) {
		return make_status(StatusCode::kInvalidArgument, "Input and bias must have same size");
	}
	if (!is_power_of_two(input.size())) {
		return make_status(StatusCode::kInvalidArgument, "Input length must be power of two");
	}
	out.resize(input.size());
	fused_fwht_bias_silu(input.data(), bias.data(), input.size(), out.data());
	return Status::OK();
}

inline Status quantize_dequantize(const std::vector<float> &input, std::vector<int8_t> &q, std::vector<float> &dq, QuantParams &params_out) {
	if (input.empty()) {
		return make_status(StatusCode::kInvalidArgument, "Input is empty");
	}
	params_out = choose_symmetric_int8_scale(input.data(), input.size());
	q.resize(input.size());
	quantize_int8(input.data(), input.size(), q.data(), params_out);
	dq.resize(input.size());
	dequantize_int8(q.data(), q.size(), params_out.scale, dq.data());
	return Status::OK();
}

// ===== reward =====
inline float reward_mse(const std::vector<float> &pred, const std::vector<float> &target) {
	if (pred.size() != target.size() || pred.empty()) return 0.0f;
	double sum = 0.0;
	for (std::size_t i = 0; i < pred.size(); ++i) {
		double d = static_cast<double>(pred[i] - target[i]);
		sum += d * d;
	}
	return static_cast<float>(sum / static_cast<double>(pred.size()));
}

inline float reward_cosine_similarity(const std::vector<float> &a, const std::vector<float> &b) {
	if (a.size() != b.size() || a.empty()) return 0.0f;
	double dot = 0.0;
	double na = 0.0;
	double nb = 0.0;
	for (std::size_t i = 0; i < a.size(); ++i) {
		dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
		na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
		nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
	}
	if (na == 0.0 || nb == 0.0) return 0.0f;
	return static_cast<float>(dot / std::sqrt(na * nb));
}

inline float reward_top1_accuracy(const std::vector<float> &logits, const std::vector<int> &labels, std::size_t num_classes) {
	if (labels.empty() || logits.size() != labels.size() * num_classes) return 0.0f;
	std::size_t correct = 0;
	for (std::size_t i = 0; i < labels.size(); ++i) {
		const float *row = logits.data() + i * num_classes;
		std::size_t argmax = 0;
		float best = row[0];
		for (std::size_t c = 1; c < num_classes; ++c) {
			if (row[c] > best) { best = row[c]; argmax = c; }
		}
		if (static_cast<std::size_t>(labels[i]) == argmax) ++correct;
	}
	return labels.empty() ? 0.0f : static_cast<float>(correct) / static_cast<float>(labels.size());
}

// ===== training =====
struct OptimizerConfig {
	float lr = 1e-2f;
};

inline void sgd_step(std::vector<float> &params, const std::vector<float> &grads, const OptimizerConfig &cfg) {
	const std::size_t n = params.size();
	for (std::size_t i = 0; i < n; ++i) {
		params[i] -= cfg.lr * grads[i];
	}
}

inline void approximate_backprop(const std::vector<float> &pred, const std::vector<float> &target, std::vector<float> &grad_out) {
	grad_out.resize(pred.size());
	for (std::size_t i = 0; i < pred.size(); ++i) {
		grad_out[i] = 2.0f * (pred[i] - target[i]) / static_cast<float>(pred.size());
	}
}

struct TrainingStepResult {
	float loss = 0.0f;
};

inline Status train_step_mse(std::vector<float> &params, const std::vector<float> &inputs, const std::vector<float> &targets,
	OptimizerConfig opt, TrainingStepResult &out) {
	if (inputs.size() != targets.size()) return make_status(StatusCode::kInvalidArgument, "inputs and targets size mismatch");
	// Dummy forward: identity prediction from params[0] scaling
	std::vector<float> pred(inputs.size());
	for (std::size_t i = 0; i < inputs.size(); ++i) pred[i] = params[0] * inputs[i];
	float loss = reward_mse(pred, targets);
	std::vector<float> grad_pred;
	approximate_backprop(pred, targets, grad_pred);
	float grad_param0 = 0.0f;
	for (std::size_t i = 0; i < inputs.size(); ++i) grad_param0 += grad_pred[i] * inputs[i];
	std::vector<float> grads = {grad_param0};
	sgd_step(params, grads, opt);
	out.loss = loss;
	return Status::OK();
}

// ===== v2.1: low-level non-temporal store helpers =====
#if defined(__AVX2__)
KLLM_INLINE void stream_store_float32_aligned(float *dst, const float *src) {
	// Use unaligned load to avoid faults if src is not 32B aligned; dst is aligned by caller logic
	__m256 v = _mm256_loadu_ps(src);
	_mm256_stream_ps(dst, v);
}
#endif

KLLM_INLINE void maybe_stream_store_range(float *dst, const float *src, std::size_t count) {
	if (!dst || !src || count == 0) return;
	const bool allow_nt = global_config().enable_pipeline_nt_stores;
	if (!allow_nt) {
		// Fallback copy
		for (std::size_t i = 0; i < count; ++i) dst[i] = src[i];
		return;
	}
#if defined(__AVX2__)
	// Use non-temporal stores where possible. Align destination to 32 bytes first.
	std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(dst);
	std::size_t prefix = 0;
	if ((addr & 31u) != 0u) {
		prefix = (32u - (addr & 31u)) / sizeof(float);
		if (prefix > count) prefix = count;
		for (std::size_t i = 0; i < prefix; ++i) dst[i] = src[i];
		dst += prefix; src += prefix; count -= prefix;
	}
	std::size_t i = 0;
	for (; i + 8 <= count; i += 8) {
		stream_store_float32_aligned(dst + i, src + i);
	}
	for (; i < count; ++i) dst[i] = src[i];
	_mm_sfence();
#else
	for (std::size_t i = 0; i < count; ++i) dst[i] = src[i];
#endif
}

// ===== v2.1: quantization manager (blockwise int8/int4) =====
struct BlockwiseQuantConfig {
	std::size_t block_size = 64; // 32–128 typical
};

inline void blockwise_quantize_int8(const float *input, std::size_t length, const BlockwiseQuantConfig &cfg,
	std::vector<int8_t> &q, std::vector<float> &scales) {
	if (!input || length == 0) { q.clear(); scales.clear(); return; }
	const std::size_t bs = (cfg.block_size == 0) ? 64 : cfg.block_size;
	const std::size_t num_blocks = (length + bs - 1) / bs;
	q.resize(length);
	scales.resize(num_blocks);
	const bool do_parallel = (length >= global_config().parallel_threshold);
	if (!do_parallel) {
		for (std::size_t b = 0; b < num_blocks; ++b) {
			const std::size_t start = b * bs;
			const std::size_t end = std::min(start + bs, length);
			float max_abs = 0.0f;
			for (std::size_t i = start; i < end; ++i) { float v = input[i]; v = v < 0.0f ? -v : v; if (v > max_abs) max_abs = v; }
			float s = (max_abs <= 1e-8f) ? 1.0f : (max_abs / 127.0f);
			if (!std::isfinite(s) || s <= 0.0f) s = 1.0f;
			scales[b] = s;
			const float inv = 1.0f / s;
			for (std::size_t i = start; i < end; ++i) {
				int vi = static_cast<int>(input[i] * inv + (input[i] >= 0.0f ? 0.5f : -0.5f));
				if (vi > 127) vi = 127; if (vi < -127) vi = -127;
				q[i] = static_cast<int8_t>(vi);
			}
		}
		return;
	}
	std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
	ThreadPool &pool = get_thread_local_pool(threads);
	for (std::size_t b = 0; b < num_blocks; ++b) {
		pool.enqueue([&, b]() {
			const std::size_t start = b * bs;
			const std::size_t end = std::min(start + bs, length);
			float max_abs = 0.0f;
			for (std::size_t i = start; i < end; ++i) { float v = input[i]; v = v < 0.0f ? -v : v; if (v > max_abs) max_abs = v; }
			float s = (max_abs <= 1e-8f) ? 1.0f : (max_abs / 127.0f);
			if (!std::isfinite(s) || s <= 0.0f) s = 1.0f;
			scales[b] = s;
			const float inv = 1.0f / s;
			for (std::size_t i = start; i < end; ++i) {
				int vi = static_cast<int>(input[i] * inv + (input[i] >= 0.0f ? 0.5f : -0.5f));
				if (vi > 127) vi = 127; if (vi < -127) vi = -127;
				q[i] = static_cast<int8_t>(vi);
			}
		});
	}
	pool.wait();
}

inline void blockwise_dequantize_int8(const int8_t *q, std::size_t length, const BlockwiseQuantConfig &cfg,
	const std::vector<float> &scales, std::vector<float> &output) {
	if (!q || length == 0) { output.clear(); return; }
	const std::size_t bs = (cfg.block_size == 0) ? 64 : cfg.block_size;
	const std::size_t num_blocks = (length + bs - 1) / bs;
	output.resize(length);
	for (std::size_t b = 0; b < num_blocks; ++b) {
		const float s = (b < scales.size() && std::isfinite(scales[b]) && scales[b] > 0.0f) ? scales[b] : 1.0f;
		const std::size_t start = b * bs;
		const std::size_t end = std::min(start + bs, length);
		for (std::size_t i = start; i < end; ++i) output[i] = static_cast<float>(q[i]) * s;
	}
}

// Int4 pack: two signed 4-bit values per byte, range [-7, 7]
inline uint8_t pack_int4_pair(int v0, int v1) {
	int a = (v0 < -7 ? -7 : (v0 > 7 ? 7 : v0));
	int b = (v1 < -7 ? -7 : (v1 > 7 ? 7 : v1));
	uint8_t lo = static_cast<uint8_t>(a & 0x0F);
	uint8_t hi = static_cast<uint8_t>((b & 0x0F) << 4);
	return static_cast<uint8_t>(lo | hi);
}

inline void unpack_int4_pair(uint8_t byte, int &v0, int &v1) {
	int lo = static_cast<int>(byte & 0x0F);
	int hi = static_cast<int>((byte >> 4) & 0x0F);
	// sign-extend 4-bit signed values (range -8..7). We clamp later to [-7,7].
	if (lo & 0x08) lo |= ~0x0F; if (hi & 0x08) hi |= ~0x0F;
	if (lo < -7) lo = -7; if (lo > 7) lo = 7;
	if (hi < -7) hi = -7; if (hi > 7) hi = 7;
	v0 = lo; v1 = hi;
}

struct BlockwiseInt4Buffer {
	std::vector<uint8_t> data; // packed
	std::vector<float> scales; // per block
	std::size_t original_length = 0;
	std::size_t block_size = 32;
};

inline void blockwise_quantize_int4(const float *input, std::size_t length, std::size_t block_size, BlockwiseInt4Buffer &out) {
	if (!input || length == 0) { out.data.clear(); out.scales.clear(); out.original_length = 0; out.block_size = block_size; return; }
	const std::size_t bs = (block_size == 0) ? 32 : block_size;
	const std::size_t num_blocks = (length + bs - 1) / bs;
	out.original_length = length; out.block_size = bs; out.scales.resize(num_blocks);
	const std::size_t packed_per_block = (bs + 1) / 2;
	out.data.resize(num_blocks * packed_per_block);
	const bool do_parallel = (length >= global_config().parallel_threshold);
	if (!do_parallel) {
		for (std::size_t b = 0; b < num_blocks; ++b) {
			const std::size_t start = b * bs;
			const std::size_t end = std::min(start + bs, length);
			float max_abs = 0.0f;
			for (std::size_t i = start; i < end; ++i) { float v = input[i]; v = v < 0.0f ? -v : v; if (v > max_abs) max_abs = v; }
			float s = (max_abs <= 1e-8f) ? 1.0f : (max_abs / 7.0f);
			if (!std::isfinite(s) || s <= 0.0f) s = 1.0f;
			out.scales[b] = s;
			const float inv = 1.0f / s;
			for (std::size_t i = start, k = 0; i < end; i += 2, ++k) {
				int v0 = static_cast<int>(input[i] * inv + (input[i] >= 0.0f ? 0.5f : -0.5f));
				int v1 = 0;
				if (i + 1 < end) v1 = static_cast<int>(input[i + 1] * inv + (input[i + 1] >= 0.0f ? 0.5f : -0.5f));
				out.data[b * packed_per_block + k] = pack_int4_pair(v0, v1);
			}
		}
		return;
	}
	std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
	ThreadPool &pool = get_thread_local_pool(threads);
	for (std::size_t b = 0; b < num_blocks; ++b) {
		pool.enqueue([&, b]() {
			const std::size_t start = b * bs;
			const std::size_t end = std::min(start + bs, length);
			float max_abs = 0.0f;
			for (std::size_t i = start; i < end; ++i) { float v = input[i]; v = v < 0.0f ? -v : v; if (v > max_abs) max_abs = v; }
			float s = (max_abs <= 1e-8f) ? 1.0f : (max_abs / 7.0f);
			if (!std::isfinite(s) || s <= 0.0f) s = 1.0f;
			out.scales[b] = s;
			const float inv = 1.0f / s;
			for (std::size_t i = start, k = 0; i < end; i += 2, ++k) {
				int v0 = static_cast<int>(input[i] * inv + (input[i] >= 0.0f ? 0.5f : -0.5f));
				int v1 = 0; if (i + 1 < end) v1 = static_cast<int>(input[i + 1] * inv + (input[i + 1] >= 0.0f ? 0.5f : -0.5f));
				out.data[b * packed_per_block + k] = pack_int4_pair(v0, v1);
			}
		});
	}
	pool.wait();
}

inline void blockwise_dequantize_int4(const BlockwiseInt4Buffer &in, std::vector<float> &out) {
	const std::size_t length = in.original_length;
	out.resize(length);
	const std::size_t bs = (in.block_size == 0) ? 32 : in.block_size;
	const std::size_t num_blocks = (length + bs - 1) / bs;
	const std::size_t packed_per_block = (bs + 1) / 2;
	for (std::size_t b = 0; b < num_blocks; ++b) {
		const float s = (b < in.scales.size() && std::isfinite(in.scales[b]) && in.scales[b] > 0.0f) ? in.scales[b] : 1.0f;
		const std::size_t start = b * bs;
		const std::size_t end = std::min(start + bs, length);
		for (std::size_t k = 0, i = start; i < end; ++k) {
			int v0, v1; unpack_int4_pair(in.data[b * packed_per_block + k], v0, v1);
			out[i++] = static_cast<float>(v0) * s;
			if (i < end) out[i++] = static_cast<float>(v1) * s;
		}
	}
}

// ===== v2.1 sketch engine (fused SignHash+Accumulate with stats) =====
struct SketchEngineV2 {
	std::size_t sketch_size = 0;
	std::size_t num_hashes = 3;
	std::uint64_t seed_base = 0x12345678abcdef00ull;
	// Outputs optional collision counts for telemetry
	void apply(const float *input, std::size_t length, float *output, std::vector<std::size_t> *collisions_out = nullptr) const {
		if (!input || !output || sketch_size == 0 || num_hashes == 0) return;
		for (std::size_t i = 0; i < sketch_size; ++i) output[i] = 0.0f;
		if (collisions_out) collisions_out->assign(sketch_size, 0);
		// Parallelize when large enough
		const std::size_t threshold = global_config().parallel_threshold;
		const bool do_parallel = (length >= threshold);
		if (!do_parallel) {
			for (std::size_t i = 0; i < length; ++i) {
				for (std::size_t h = 0; h < num_hashes; ++h) {
					const std::uint64_t mix = splitmix64(static_cast<std::uint64_t>(i) ^ (seed_base + static_cast<std::uint64_t>(h) * 0x9e3779b97f4a7c15ull));
					const std::size_t bucket = static_cast<std::size_t>(mix % static_cast<std::uint64_t>(sketch_size));
					const float sign = ((mix >> 63) == 0ull) ? 1.0f : -1.0f;
					const float before = output[bucket];
					output[bucket] = before + sign * input[i];
					if (collisions_out && before != 0.0f) { ++(*collisions_out)[bucket]; }
				}
			}
			// Collision-aware scaling (simple): scale buckets with high collision counts down slightly
			if (collisions_out) {
				for (std::size_t b = 0; b < sketch_size; ++b) {
					std::size_t c = (*collisions_out)[b];
					if (c > 0) { float scale = 1.0f / (1.0f + static_cast<float>(c)); output[b] *= scale; }
				}
			}
			return;
		}
		// Parallel path: thread-local accumulators, final reduction
		std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		ThreadPool &pool = get_thread_local_pool(threads);
		std::vector<std::vector<float>> local(threads, std::vector<float>(sketch_size, 0.0f));
		std::vector<std::vector<std::size_t>> local_coll;
		if (collisions_out) local_coll.assign(threads, std::vector<std::size_t>(sketch_size, 0));
		const std::size_t chunk = (length + threads - 1) / threads;
		for (std::size_t t = 0; t < threads; ++t) {
			const std::size_t start = t * chunk;
			if (start >= length) continue;
			const std::size_t end = std::min(start + chunk, length);
			pool.enqueue([&, t, start, end]() {
				float *out_t = local[t].data();
				std::size_t *coll_t = collisions_out ? local_coll[t].data() : nullptr;
				for (std::size_t i = start; i < end; ++i) {
					for (std::size_t h = 0; h < num_hashes; ++h) {
						const std::uint64_t mix = splitmix64(static_cast<std::uint64_t>(i) ^ (seed_base + static_cast<std::uint64_t>(h) * 0x9e3779b97f4a7c15ull));
						const std::size_t bucket = static_cast<std::size_t>(mix % static_cast<std::uint64_t>(sketch_size));
						const float sign = ((mix >> 63) == 0ull) ? 1.0f : -1.0f;
						const float before = out_t[bucket];
						out_t[bucket] = before + sign * input[i];
						if (coll_t && before != 0.0f) { coll_t[bucket] += 1; }
					}
				}
			});
		}
		pool.wait();
		// Reduce
		for (std::size_t b = 0; b < sketch_size; ++b) {
			float acc = 0.0f;
			for (std::size_t t = 0; t < threads; ++t) acc += local[t][b];
			output[b] = acc;
		}
		if (collisions_out) {
			for (std::size_t b = 0; b < sketch_size; ++b) {
				std::size_t c = 0;
				for (std::size_t t = 0; t < threads; ++t) c += local_coll[t][b];
				(*collisions_out)[b] = c;
			}
			for (std::size_t b = 0; b < sketch_size; ++b) { if ((*collisions_out)[b] > 0) { output[b] *= 1.0f / (1.0f + static_cast<float>((*collisions_out)[b])); } }
		}
	}
private:
};

// ===== v2.1 routing engine (two-level router) =====
struct RoutingSelection {
	std::vector<std::size_t> indices; // selected indices across all buckets
};

struct RoutingEngineV2 {
	std::size_t bucket_size = 256;
	std::size_t top_k_within_bucket = 4;
	bool stable = true; // deterministic ordering
	void route(const float *scores, std::size_t length, RoutingSelection &out) const {
		out.indices.clear();
		if (!scores || length == 0 || bucket_size == 0) return;
		const std::size_t num_buckets = (length + bucket_size - 1) / bucket_size;
		std::vector<std::vector<std::size_t>> per_bucket(num_buckets);
		const bool do_parallel = (length >= global_config().parallel_threshold);
		if (do_parallel) {
			std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
			ThreadPool &pool = get_thread_local_pool(threads);
			for (std::size_t b = 0; b < num_buckets; ++b) {
				pool.enqueue([&, b]() {
					const std::size_t start = b * bucket_size;
					const std::size_t end = std::min(start + bucket_size, length);
					std::vector<std::pair<float, std::size_t>> heap;
					heap.reserve(end - start);
					for (std::size_t i = start; i < end; ++i) {
						float s = scores[i]; s = s < 0.0f ? -s : s;
						heap.emplace_back(s, i);
					}
					const std::size_t k = std::min<std::size_t>(top_k_within_bucket, heap.size());
					std::partial_sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(k), heap.end(),
						[](const auto &a, const auto &b){ return a.first > b.first; });
					per_bucket[b].reserve(k);
					for (std::size_t i = 0; i < k; ++i) per_bucket[b].push_back(heap[i].second);
				});
			}
			pool.wait();
		} else {
			for (std::size_t b = 0; b < num_buckets; ++b) {
				const std::size_t start = b * bucket_size;
				const std::size_t end = std::min(start + bucket_size, length);
				std::vector<std::pair<float, std::size_t>> heap;
				heap.reserve(end - start);
				for (std::size_t i = start; i < end; ++i) { float s = scores[i]; s = s < 0.0f ? -s : s; heap.emplace_back(s, i); }
				const std::size_t k = std::min<std::size_t>(top_k_within_bucket, heap.size());
				std::partial_sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(k), heap.end(),
					[](const auto &a, const auto &b){ return a.first > b.first; });
				per_bucket[b].reserve(k);
				for (std::size_t i = 0; i < k; ++i) per_bucket[b].push_back(heap[i].second);
			}
		}
		// Flatten
		std::size_t total = 0; for (const auto &v : per_bucket) total += v.size();
		out.indices.resize(total);
		std::size_t pos = 0;
		for (std::size_t b = 0; b < num_buckets; ++b) {
			for (std::size_t idx : per_bucket[b]) out.indices[pos++] = idx;
		}
		if (stable) {
			std::stable_sort(out.indices.begin(), out.indices.end());
		}
	}
};

// ===== v2.1 execution pipeline (streaming slabs) =====
struct PipelineTelemetry {
	double ms_total = 0.0;
	double ms_stage0 = 0.0;
	double ms_stage1 = 0.0;
	double ms_stage2 = 0.0;
	std::size_t slabs_processed = 0;
};

enum class PointwiseOp { kIdentity = 0, kRelu = 1, kGelu = 2, kSilu = 3 };

inline void apply_pointwise(PointwiseOp op, float *data, std::size_t length) {
	if (!data) return;
	switch (op) {
		case PointwiseOp::kIdentity: return;
		case PointwiseOp::kRelu:
			for (std::size_t i = 0; i < length; ++i) if (data[i] < 0.0f) data[i] = 0.0f; return;
		case PointwiseOp::kGelu:
			for (std::size_t i = 0; i < length; ++i) {
				float x = data[i]; float t = x * (0.7978845608028654f) * (1.0f + 0.044715f * x * x); data[i] = 0.5f * x * (1.0f + std::tanh(t));
			}
			return;
		case PointwiseOp::kSilu:
			for (std::size_t i = 0; i < length; ++i) { float x = data[i]; float s = 1.0f / (1.0f + std::exp(-x)); data[i] = x * s; }
			return;
	}
}

struct PipelineConfigV21 {
	std::size_t sketch_size = 0;
	PointwiseOp pointwise = PointwiseOp::kRelu;
	bool use_int8 = true;
	bool use_int4 = false; // if true, overrides int8
	BlockwiseQuantConfig qcfg;
};

inline Status pipeline_transform_sketch_route_quantize_v21(const std::vector<float> &input, std::vector<uint8_t> &q4_out,
	std::vector<int8_t> &q8_out, std::vector<float> &scales_out, PipelineTelemetry &telemetry, const PipelineConfigV21 &pcfg) {
	if (input.empty()) return make_status(StatusCode::kInvalidArgument, "pipeline: input empty");
	if (!is_power_of_two(input.size())) return make_status(StatusCode::kInvalidArgument, "pipeline: length must be power of two");
	if (pcfg.sketch_size == 0) return make_status(StatusCode::kInvalidArgument, "pipeline: sketch_size must be > 0");
	const std::size_t n = input.size();
	const std::size_t slab_bytes = global_config().large_slab_bytes;
	const std::size_t slab_floats = std::max<std::size_t>(1, slab_bytes / sizeof(float));
	const std::size_t slab_len = std::min<std::size_t>(n, slab_floats);
	const std::size_t inflight = std::max<std::size_t>(1, std::min<std::size_t>(global_config().max_inflight_slabs, 3));
	// allocate ring buffers (reserve for reuse)
	std::vector<std::vector<float>> ring_input(inflight);
	std::vector<std::vector<float>> ring_transformed(inflight);
	std::vector<std::vector<float>> ring_sketch(inflight);
	for (std::size_t r = 0; r < inflight; ++r) { ring_input[r].reserve(slab_len); ring_transformed[r].reserve(slab_len); ring_sketch[r].reserve(pcfg.sketch_size); }
	SketchEngineV2 sk{}; sk.sketch_size = pcfg.sketch_size; sk.num_hashes = global_config().sketch_num_hashes;
	RoutingEngineV2 router{}; router.bucket_size = global_config().routing_bucket_size; router.top_k_within_bucket = 4; router.stable = global_config().deterministic;
	Profiler prof;
	ScopeTimer total_timer(prof, "pipeline_total");
	std::size_t processed = 0; telemetry.slabs_processed = 0;
	while (processed < n) {
		const std::size_t slab_index = telemetry.slabs_processed % inflight;
		const std::size_t chunk = std::min<std::size_t>(slab_len, n - processed);
		// S0: load into ring buffer (memcpy to reuse capacity)
		{
			ScopeTimer t0(prof, "stage0");
			ring_input[slab_index].resize(chunk);
			std::memcpy(ring_input[slab_index].data(), input.data() + static_cast<std::ptrdiff_t>(processed), chunk * sizeof(float));
		}
		// S1: Transform + Sketch
		{
			ScopeTimer t1(prof, "stage1");
			ring_transformed[slab_index] = ring_input[slab_index];
			if (chunk >= global_config().parallel_threshold) {
				std::size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
				ThreadPool &pool = get_thread_local_pool(threads);
				fwht_inplace_parallel(ring_transformed[slab_index].data(), ring_transformed[slab_index].size(), pool);
			} else {
				fwht_inplace(ring_transformed[slab_index].data(), ring_transformed[slab_index].size());
			}
			ring_sketch[slab_index].resize(pcfg.sketch_size);
			sk.apply(ring_transformed[slab_index].data(), ring_transformed[slab_index].size(), ring_sketch[slab_index].data(), nullptr);
			apply_pointwise(pcfg.pointwise, ring_sketch[slab_index].data(), ring_sketch[slab_index].size());
		}
		// S2: Route + Quantize
		{
			ScopeTimer t2(prof, "stage2");
			RoutingSelection sel{}; router.route(ring_sketch[slab_index].data(), ring_sketch[slab_index].size(), sel);
			// Gather selected into a contiguous buffer
			std::vector<float> selected(sel.indices.size());
			for (std::size_t i = 0; i < sel.indices.size(); ++i) selected[i] = ring_sketch[slab_index][sel.indices[i]];
			if (pcfg.use_int4) {
				BlockwiseInt4Buffer tmp{}; blockwise_quantize_int4(selected.data(), selected.size(), 32, tmp);
				// append to q4_out and scales_out
				const std::size_t prev_bytes = q4_out.size();
				q4_out.resize(prev_bytes + tmp.data.size());
				if (!tmp.data.empty()) std::memcpy(q4_out.data() + static_cast<std::ptrdiff_t>(prev_bytes), tmp.data.data(), tmp.data.size());
				const std::size_t prev_sc = scales_out.size();
				scales_out.resize(prev_sc + tmp.scales.size());
				for (std::size_t i = 0; i < tmp.scales.size(); ++i) scales_out[prev_sc + i] = tmp.scales[i];
			} else if (pcfg.use_int8) {
				std::vector<float> sc; std::vector<int8_t> qtmp; BlockwiseQuantConfig qcfg = pcfg.qcfg; if (qcfg.block_size == 0) qcfg.block_size = 64;
				blockwise_quantize_int8(selected.data(), selected.size(), qcfg, qtmp, sc);
				// append
				const std::size_t prev = q8_out.size(); q8_out.resize(prev + qtmp.size());
				for (std::size_t i = 0; i < qtmp.size(); ++i) q8_out[prev + i] = qtmp[i];
				const std::size_t prev_sc = scales_out.size(); scales_out.resize(prev_sc + sc.size());
				for (std::size_t i = 0; i < sc.size(); ++i) scales_out[prev_sc + i] = sc[i];
			}
		}
		processed += chunk;
		telemetry.slabs_processed += 1;
	}
	// collect timings
	telemetry.ms_total = prof.get("pipeline_total").ms;
	telemetry.ms_stage0 = prof.get("stage0").ms;
	telemetry.ms_stage1 = prof.get("stage1").ms;
	telemetry.ms_stage2 = prof.get("stage2").ms;
	return Status::OK();
}

// ===== v2.1 scheduler & cost model =====
struct CostModelV21Metrics {
	double cycles = 0.0;
	double bytes_moved = 0.0;
	double reuse_ratio = 1.0;
	double tlb_pressure = 0.0;
};

enum class ExecutionMode { kSmallN = 0, kLargeN = 1 };

inline ExecutionMode choose_mode_v21(std::size_t n, const CostModelV21Metrics &m) {
	if (n > (1u << 16)) { // > 64K
		if (m.reuse_ratio < 1.1) return ExecutionMode::kLargeN;
	}
	return (n >= global_config().parallel_threshold) ? ExecutionMode::kLargeN : ExecutionMode::kSmallN;
}

// Public wrapper selecting small vs large pipeline. SmallN reuses fused kernels, LargeN runs streaming pipeline.
inline Status transform_sketch_route_quantize_auto(const std::vector<float> &input, std::size_t sketch_size,
	std::vector<uint8_t> &q4_out, std::vector<int8_t> &q8_out, std::vector<float> &scales_out,
	PipelineTelemetry &telemetry, PointwiseOp pointwise = PointwiseOp::kRelu, bool prefer_int4 = false) {
	CostModelV21Metrics m{}; m.reuse_ratio = 1.0; // heuristic for transforms
	ExecutionMode mode = choose_mode_v21(input.size(), m);
	PipelineConfigV21 pc{}; pc.sketch_size = sketch_size; pc.pointwise = pointwise; pc.use_int4 = prefer_int4; pc.use_int8 = !prefer_int4;
	if (mode == ExecutionMode::kSmallN) {
		// one-shot: FWHT then sketch then pointwise then quantize
		std::vector<float> tmp = input;
		fwht_inplace(tmp.data(), tmp.size());
		SketchEngineV2 sk{}; sk.sketch_size = sketch_size; sk.num_hashes = global_config().sketch_num_hashes;
		std::vector<float> skv(sketch_size);
		sk.apply(tmp.data(), tmp.size(), skv.data(), nullptr);
		apply_pointwise(pointwise, skv.data(), skv.size());
		Profiler prof; ScopeTimer total(prof, "pipeline_total");
		if (prefer_int4) {
			BlockwiseInt4Buffer buf{}; blockwise_quantize_int4(skv.data(), skv.size(), 32, buf);
			q4_out = std::move(buf.data); scales_out = std::move(buf.scales); q8_out.clear();
		} else {
			BlockwiseQuantConfig qcfg{}; std::vector<float> sc; std::vector<int8_t> qtmp; blockwise_quantize_int8(skv.data(), skv.size(), qcfg, qtmp, sc);
			q8_out = std::move(qtmp); scales_out = std::move(sc); q4_out.clear();
		}
		telemetry.ms_total = prof.get("pipeline_total").ms; telemetry.slabs_processed = 1;
		return Status::OK();
	}
	return pipeline_transform_sketch_route_quantize_v21(input, q4_out, q8_out, scales_out, telemetry, pc);
}

// ===== v2.1 training: reversible block (stub) & sketch-backprop approx =====
struct ReversibleBlockConfig { std::size_t width = 0; };

inline void reversible_block_forward(const std::vector<float> &x, std::vector<float> &y) {
	y = x; // identity stub to keep API footprint; extend later
}

inline void sketch_backprop_approx(const std::vector<float> &grad_out, std::vector<float> &grad_in) {
	grad_in = grad_out; // passthrough stub
}

// ===== Public API v2.1 =====
inline Status run_pipeline_v21_to_int8(const std::vector<float> &input, std::size_t sketch_size, std::vector<int8_t> &q8, std::vector<float> &scales, PipelineTelemetry &telemetry, PointwiseOp pointwise = PointwiseOp::kRelu) {
	std::vector<uint8_t> q4_dummy; return transform_sketch_route_quantize_auto(input, sketch_size, q4_dummy, q8, scales, telemetry, pointwise, false);
}

inline Status run_pipeline_v21_to_int4(const std::vector<float> &input, std::size_t sketch_size, std::vector<uint8_t> &q4, std::vector<float> &scales, PipelineTelemetry &telemetry, PointwiseOp pointwise = PointwiseOp::kRelu) {
	std::vector<int8_t> q8_dummy; return transform_sketch_route_quantize_auto(input, sketch_size, q4, q8_dummy, scales, telemetry, pointwise, true);
}

// ===== nn/autograd/optim/data/trainer (minimal, header-only) =====
namespace nn {
	struct Value; using ValuePtr = std::shared_ptr<Value>;
			struct Value : public std::enable_shared_from_this<Value> {
			std::vector<float> values, grad; std::vector<std::size_t> shape; bool requires_grad = false;
			std::vector<std::shared_ptr<Value>> parents; std::function<void()> backward_fn;
			static ValuePtr create(const std::vector<std::size_t> &s, bool rg) { ValuePtr v = std::make_shared<Value>(); v->shape = s; std::size_t n=1; for (auto d:s) n*=d; v->values.assign(n,0.0f); if(rg){ v->grad.assign(n,0.0f); } else { v->grad.clear(); } v->requires_grad = rg; return v; }
			std::size_t numel() const { std::size_t n=1; for (auto d:shape) n*=d; return n; }
			void zero_grad(){ for(float &g:grad) g=0.0f; }
			float * data(){ return values.empty()? nullptr: values.data(); }
			const float * data() const { return values.empty()? nullptr: values.data(); }
			void backward(){ if (grad.empty()) grad.assign(values.size(),0.0f); if (values.size()==1) grad[0]=1.0f; std::vector<ValuePtr> topo; std::unordered_map<Value*,int> vis; std::function<void(ValuePtr)> dfs=[&](ValuePtr u){ if(vis[u.get()])return; vis[u.get()]=1; for(auto &p:u->parents){ dfs(p);} topo.push_back(u);}; dfs(shared_from_this()); for(auto &node: topo){ if(node->requires_grad && node->grad.size()!=node->values.size()){ node->grad.assign(node->values.size(), 0.0f); } } for(auto it=topo.rbegin(); it!=topo.rend(); ++it){ if((*it)->backward_fn) (*it)->backward_fn(); } }
		};
	inline bool same(const std::vector<std::size_t>&a,const std::vector<std::size_t>&b){ if(a.size()!=b.size()) return false; for(size_t i=0;i<a.size();++i) if(a[i]!=b[i]) return false; return true; }
	inline ValuePtr tensor(const std::vector<float>&v,const std::vector<std::size_t>&s,bool rg=false){ auto t=Value::create(s,rg); if(t->values.size()==v.size()) t->values=v; return t; }
	inline ValuePtr zeros(const std::vector<std::size_t>&s,bool rg=false){ return Value::create(s,rg);} 
	inline ValuePtr ones(const std::vector<std::size_t>&s,bool rg=false){ auto t=Value::create(s,rg); for(float &x:t->values) x=1.0f; return t; }
	inline ValuePtr full(const std::vector<std::size_t>&s,float v,bool rg=false){ auto t=Value::create(s,rg); for(float &x:t->values) x=v; return t; }
	inline ValuePtr randu(const std::vector<std::size_t>&s,unsigned seed=123,bool rg=false){ std::mt19937 rng(seed); std::uniform_real_distribution<float> ud(0.f,1.f); auto t=Value::create(s,rg); for(float &x:t->values) x=ud(rng); return t; }
	inline ValuePtr randn(const std::vector<std::size_t>&s,unsigned seed=123,float stddev=0.02f,bool rg=false){ std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f,stddev); auto t=Value::create(s,rg); for(float &x:t->values) x=nd(rng); return t; }
	inline void xavier_uniform_(Value &w, unsigned seed=123){ if(w.shape.size()<2) return; float fan_in=float(w.shape[w.shape.size()-2]); float fan_out=float(w.shape[w.shape.size()-1]); float limit = std::sqrt(6.0f/(fan_in+fan_out)); std::mt19937 rng(seed); std::uniform_real_distribution<float> ud(-limit, limit); for(float &v:w.values) v = ud(rng); }
	inline void he_uniform_(Value &w, unsigned seed=123){ if(w.shape.size()<2) return; float fan_in=float(w.shape[w.shape.size()-2]); float limit = std::sqrt(6.0f/fan_in); std::mt19937 rng(seed); std::uniform_real_distribution<float> ud(-limit, limit); for(float &v:w.values) v = ud(rng); }
	inline ValuePtr add(const ValuePtr&a,const ValuePtr&b){ if(!same(a->shape,b->shape)) return nullptr; auto o=Value::create(a->shape,a->requires_grad||b->requires_grad); for(size_t i=0;i<o->values.size();++i) o->values[i]=a->values[i]+b->values[i]; o->parents={a,b}; Value *ap=a.get(), *bp=b.get(), *op=o.get(); o->backward_fn=[op,ap,bp](){ if(ap->requires_grad) for(size_t i=0;i<op->values.size();++i) ap->grad[i]+=op->grad[i]; if(bp->requires_grad) for(size_t i=0;i<op->values.size();++i) bp->grad[i]+=op->grad[i]; }; return o; }
	inline ValuePtr mul(const ValuePtr&a,const ValuePtr&b){ if(!same(a->shape,b->shape)) return nullptr; auto o=Value::create(a->shape,a->requires_grad||b->requires_grad); for(size_t i=0;i<o->values.size();++i) o->values[i]=a->values[i]*b->values[i]; o->parents={a,b}; Value *ap=a.get(), *bp=b.get(), *op=o.get(); o->backward_fn=[op,ap,bp](){ if(ap->requires_grad) for(size_t i=0;i<op->values.size();++i) ap->grad[i]+=op->grad[i]*bp->values[i]; if(bp->requires_grad) for(size_t i=0;i<op->values.size();++i) bp->grad[i]+=op->grad[i]*ap->values[i]; }; return o; }
	inline ValuePtr relu(const ValuePtr&x){ auto o=Value::create(x->shape,x->requires_grad); for(size_t i=0;i<o->values.size();++i) o->values[i]=x->values[i]<0.f?0.f:x->values[i]; o->parents={x}; Value *xp=x.get(), *op=o.get(); o->backward_fn=[op,xp](){ if(!xp->requires_grad) return; for(size_t i=0;i<op->values.size();++i) xp->grad[i]+=op->grad[i]*(xp->values[i]>0.f?1.f:0.f);} ; return o; }
	inline ValuePtr leaky_relu(const ValuePtr&x,float alpha=0.01f){ auto o=Value::create(x->shape,x->requires_grad); for(size_t i=0;i<o->values.size();++i){ float v=x->values[i]; o->values[i]= (v>0.f)? v : alpha*v; } o->parents={x}; Value *xp=x.get(), *op=o.get(); o->backward_fn=[op,xp,alpha](){ if(!xp->requires_grad) return; for(size_t i=0;i<op->values.size();++i){ float v=xp->values[i]; xp->grad[i]+=op->grad[i]*((v>0.f)?1.f:alpha);} }; return o; }
	inline ValuePtr elu(const ValuePtr&x,float alpha=1.0f){ auto o=Value::create(x->shape,x->requires_grad); for(size_t i=0;i<o->values.size();++i){ float v=x->values[i]; o->values[i] = (v>=0.f)? v : alpha*(std::exp(v)-1.f); } o->parents={x}; Value *xp=x.get(), *op=o.get(); o->backward_fn=[op,xp,alpha](){ if(!xp->requires_grad) return; for(size_t i=0;i<op->values.size();++i){ float v=xp->values[i]; float der = (v>=0.f)? 1.f : (alpha*std::exp(v)); xp->grad[i]+=op->grad[i]*der; } }; return o; }
	inline ValuePtr selu(const ValuePtr&x){ const float lambda=1.0507009873554805f, alpha=1.6732632423543772f; auto o=Value::create(x->shape,x->requires_grad); for(size_t i=0;i<o->values.size();++i){ float v=x->values[i]; o->values[i] = lambda * ((v>=0.f)? v : alpha*(std::exp(v)-1.f)); } o->parents={x}; Value *xp=x.get(), *op=o.get(); o->backward_fn=[op,xp,lambda,alpha](){ if(!xp->requires_grad) return; for(size_t i=0;i<op->values.size();++i){ float v=xp->values[i]; float der = lambda * ((v>=0.f)? 1.f : (alpha*std::exp(v))); xp->grad[i]+=op->grad[i]*der; } }; return o; }
	inline ValuePtr gelu(const ValuePtr&x){ auto o=Value::create(x->shape,x->requires_grad); for(size_t i=0;i<o->values.size();++i){ float v=x->values[i]; float t=v*(0.7978845608028654f)*(1.f+0.044715f*v*v); o->values[i]=0.5f*v*(1.f+std::tanh(t)); } o->parents={x}; Value *xp=x.get(), *op=o.get(); o->backward_fn=[op,xp](){ if(!xp->requires_grad) return; for(size_t i=0;i<op->values.size();++i){ float v=xp->values[i]; float th=std::tanh(0.7978845608028654f*v*(1.f+0.044715f*v*v)); float tt=0.7978845608028654f*(1.f+0.134145f*v*v); float dy=0.5f*(1.f+th)+0.5f*v*(1.f-th*th)*tt; xp->grad[i]+=op->grad[i]*dy; } }; return o; }
	inline ValuePtr matmul(const ValuePtr&A,const ValuePtr&B){
		if(A->shape.size()!=2||B->shape.size()!=2) return nullptr; size_t M=A->shape[0],K=A->shape[1],K2=B->shape[0],N=B->shape[1]; if(K!=K2) return nullptr;
		auto o=Value::create({M,N},A->requires_grad||B->requires_grad);
		float * C = o->values.data();
		const float * a = A->values.data();
		const float * b = B->values.data();
		const bool blocked = global_config().enable_matmul_blocked;
		const std::size_t BM = std::max<std::size_t>(1, global_config().matmul_block_m);
		const std::size_t BN = std::max<std::size_t>(1, global_config().matmul_block_n);
		const std::size_t BK = std::max<std::size_t>(1, global_config().matmul_block_k);
		auto kernel_tile = [&](size_t i0, size_t j0, size_t mlim, size_t nlim, size_t k0, size_t klim){
			for(size_t i=i0;i<mlim;++i){
				float * ci = C + i*N;
				const float * ai = a + i*K;
				std::size_t j = j0;
#if defined(__AVX2__)
				if (global_config().enable_simd) {
					for(; j+16<=nlim; j+=16){
						__m256 sum0=_mm256_loadu_ps(ci+j); __m256 sum1=_mm256_loadu_ps(ci+j+8);
						for(size_t k=k0;k<klim;++k){ __m256 av=_mm256_set1_ps(ai[k]); const float * bj=b+k*N+j; __m256 bv0=_mm256_loadu_ps(bj); __m256 bv1=_mm256_loadu_ps(bj+8); sum0=_mm256_fmadd_ps(av,bv0,sum0); sum1=_mm256_fmadd_ps(av,bv1,sum1);} _mm256_storeu_ps(ci+j,sum0); _mm256_storeu_ps(ci+j+8,sum1);
					}
					for(; j+8<=nlim; j+=8){ __m256 sum=_mm256_loadu_ps(ci+j); for(size_t k=k0;k<klim;++k){ __m256 av=_mm256_set1_ps(ai[k]); __m256 bv=_mm256_loadu_ps(b+k*N+j); sum=_mm256_fmadd_ps(av,bv,sum);} _mm256_storeu_ps(ci+j,sum); }
				}
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
				if (global_config().enable_simd) {
					for(; j+8<=nlim; j+=8){ float32x4_t s0=vld1q_f32(ci+j), s1=vld1q_f32(ci+j+4); for(size_t k=k0;k<klim;++k){ float32x4_t av=vdupq_n_f32(ai[k]); const float * bj=b+k*N+j; float32x4_t bv0=vld1q_f32(bj), bv1=vld1q_f32(bj+4); s0=vmlaq_f32(s0,av,bv0); s1=vmlaq_f32(s1,av,bv1);} vst1q_f32(ci+j,s0); vst1q_f32(ci+j+4,s1);} for(; j+4<=nlim; j+=4){ float32x4_t s=vld1q_f32(ci+j); for(size_t k=k0;k<klim;++k){ float32x4_t av=vdupq_n_f32(ai[k]); float32x4_t bv=vld1q_f32(b+k*N+j); s=vmlaq_f32(s,av,bv);} vst1q_f32(ci+j,s);} }
#endif
				for(; j<nlim; ++j){ float acc=ci[j]; for(size_t k=k0;k<klim;++k) acc += ai[k]*b[k*N + j]; ci[j]=acc; }
			}
		};
		if (blocked && (M*N >= global_config().parallel_threshold)) {
			size_t threads = global_config().num_threads ? global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
			ThreadPool &pool = get_thread_local_pool(threads);
			for(size_t i0=0;i0<M;i0+=BM){ size_t mlim=std::min(M,i0+BM);
				pool.enqueue([=](){
					for(size_t k0=0;k0<K;k0+=BK){ size_t klim=std::min(K,k0+BK);
						for(size_t j0=0;j0<N;j0+=BN){ size_t nlim=std::min(N,j0+BN);
							kernel_tile(i0,j0,mlim,nlim,k0,klim);
						}
					}
				});
			}
			pool.wait();
		} else if (blocked) {
			for(size_t i0=0;i0<M;i0+=BM){ size_t mlim=std::min(M,i0+BM);
				for(size_t k0=0;k0<K;k0+=BK){ size_t klim=std::min(K,k0+BK);
					for(size_t j0=0;j0<N;j0+=BN){ size_t nlim=std::min(N,j0+BN);
						kernel_tile(i0,j0,mlim,nlim,k0,klim);
					}
				}
			}
		} else {
			// Non-blocked accumulation
			for (size_t i = 0; i < M; ++i) {
				float * ci = C + i*N;
				const float * ai = a + i*K;
				for (size_t k = 0; k < K; ++k) {
					const float aik = ai[k];
					const float * bk = b + k*N;
					std::size_t j = 0;
#if defined(__AVX2__)
					if (global_config().enable_simd) {
						for (; j + 16 <= N; j += 16) { __m256 av=_mm256_set1_ps(aik); __m256 s0=_mm256_loadu_ps(ci+j); __m256 s1=_mm256_loadu_ps(ci+j+8); __m256 bv0=_mm256_loadu_ps(bk+j); __m256 bv1=_mm256_loadu_ps(bk+j+8); s0=_mm256_fmadd_ps(av,bv0,s0); s1=_mm256_fmadd_ps(av,bv1,s1); _mm256_storeu_ps(ci+j,s0); _mm256_storeu_ps(ci+j+8,s1);} for (; j + 8 <= N; j += 8) { __m256 av=_mm256_set1_ps(aik); __m256 s=_mm256_loadu_ps(ci+j); __m256 bv=_mm256_loadu_ps(bk+j); s=_mm256_fmadd_ps(av,bv,s); _mm256_storeu_ps(ci+j,s);} }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
					if (global_config().enable_simd) { for (; j + 8 <= N; j += 8) { float32x4_t av=vdupq_n_f32(aik); float32x4_t s0=vld1q_f32(ci+j), s1=vld1q_f32(ci+j+4); float32x4_t bv0=vld1q_f32(bk+j), bv1=vld1q_f32(bk+j+4); s0=vmlaq_f32(s0,av,bv0); s1=vmlaq_f32(s1,av,bv1); vst1q_f32(ci+j,s0); vst1q_f32(ci+j+4,s1);} for (; j + 4 <= N; j += 4) { float32x4_t av=vdupq_n_f32(aik); float32x4_t s=vld1q_f32(ci+j); float32x4_t bv=vld1q_f32(bk+j); s=vmlaq_f32(s,av,bv); vst1q_f32(ci+j,s);} }
#endif
					for (; j < N; ++j) ci[j] += aik * bk[j];
				}
			}
		}
		o->parents={A,B}; Value *Ap=A.get(), *Bp=B.get(), *op=o.get(); o->backward_fn=[op,Ap,Bp,M,K,N](){ if(Ap->requires_grad){ for(size_t i=0;i<M;++i){ for(size_t k=0;k<K;++k){ float acc=0.f; for(size_t j=0;j<N;++j) acc+=op->grad[i*N+j]*Bp->values[k*N+j]; Ap->grad[i*K+k]+=acc; }}} if(Bp->requires_grad){ for(size_t k=0;k<K;++k){ for(size_t j=0;j<N;++j){ float acc=0.f; for(size_t i=0;i<M;++i) acc+=Ap->values[i*K+k]*op->grad[i*N+j]; Bp->grad[k*N+j]+=acc; }}} };
		return o;
	}
	inline ValuePtr add_bias(const ValuePtr&x,const ValuePtr&b){ if(x->shape.size()!=2||b->shape.size()!=1) return nullptr; size_t M=x->shape[0],N=x->shape[1]; if(b->shape[0]!=N) return nullptr; auto o=Value::create(x->shape,x->requires_grad||b->requires_grad); for(size_t i=0;i<M;++i) for(size_t j=0;j<N;++j) o->values[i*N+j]=x->values[i*N+j]+b->values[j]; o->parents={x,b}; Value *Xp=x.get(), *Bp=b.get(), *op=o.get(); o->backward_fn=[op,Xp,Bp,M,N](){ if(Xp->requires_grad) for(size_t i=0;i<M*N;++i) Xp->grad[i]+=op->grad[i]; if(Bp->requires_grad) for(size_t j=0;j<N;++j){ float acc=0.f; for(size_t i=0;i<M;++i) acc+=op->grad[i*N+j]; Bp->grad[j]+=acc; } }; return o; }
	inline ValuePtr softmax_lastdim(const ValuePtr&x){ if(x->shape.size()!=2) return nullptr; size_t B=x->shape[0],C=x->shape[1]; auto o=Value::create(x->shape,x->requires_grad); for(size_t i=0;i<B;++i){ float maxv=x->values[i*C]; for(size_t c=1;c<C;++c) maxv=std::max(maxv,x->values[i*C+c]); float sum=0.f; for(size_t c=0;c<C;++c){ float e=std::exp(x->values[i*C+c]-maxv); o->values[i*C+c]=e; sum+=e; } for(size_t c=0;c<C;++c) o->values[i*C+c]/=(sum==0.f?1.f:sum);} o->parents={x}; Value *Xp=x.get(), *op=o.get(); o->backward_fn=[op,Xp,B,C](){ if(!Xp->requires_grad) return; for(size_t i=0;i<B;++i){ float dot=0.f; for(size_t c=0;c<C;++c) dot+=op->grad[i*C+c]*op->values[i*C+c]; for(size_t c=0;c<C;++c) Xp->grad[i*C+c]+=op->values[i*C+c]*(op->grad[i*C+c]-dot); } }; return o; }
	inline ValuePtr mse_loss(const ValuePtr&pred,const ValuePtr&target){ if(!same(pred->shape,target->shape)) return nullptr; auto o=Value::create({1},pred->requires_grad||target->requires_grad); size_t n=pred->numel(); double acc=0.0; for(size_t i=0;i<n;++i){ double d=double(pred->values[i])-double(target->values[i]); acc+=d*d; } o->values[0]=float(acc/double(n)); o->parents={pred,target}; Value *Pp=pred.get(), *Tp=target.get(), *op=o.get(); o->backward_fn=[op,Pp,Tp,n](){ float g=op->grad[0]*(2.f/float(n)); if(Pp->requires_grad) for(size_t i=0;i<n;++i) Pp->grad[i]+=g*(Pp->values[i]-Tp->values[i]); if(Tp->requires_grad) for(size_t i=0;i<n;++i) Tp->grad[i]+=g*(Tp->values[i]-Pp->values[i]); }; return o; }
	inline ValuePtr cross_entropy_logits(const ValuePtr&logits,const std::vector<int>&labels){ if(logits->shape.size()!=2) return nullptr; size_t B=logits->shape[0],C=logits->shape[1]; if(labels.size()!=B) return nullptr; auto o=Value::create({1},logits->requires_grad); std::vector<float> sm(B*C); double loss=0.0; for(size_t i=0;i<B;++i){ float maxv=logits->values[i*C]; for(size_t c=1;c<C;++c) maxv=std::max(maxv,logits->values[i*C+c]); float sum=0.f; for(size_t c=0;c<C;++c){ float e=std::exp(logits->values[i*C+c]-maxv); sm[i*C+c]=e; sum+=e; } for(size_t c=0;c<C;++c) sm[i*C+c]/=(sum==0.f?1.f:sum); int y=labels[i]<0?0:(labels[i]>=int(C)?int(C)-1:labels[i]); float p=sm[i*C+size_t(y)]; loss+=-std::log(p<=1e-12f?1e-12f:p);} o->values[0]=float(loss/double(B)); o->parents={logits}; Value *Lp=logits.get(), *op=o.get(); o->backward_fn=[op,Lp,sm,B,C,labels](){ if(!Lp->requires_grad) return; float g=op->grad[0]/float(B); for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c) Lp->grad[i*C+c]+=g*sm[i*C+c]; int y=labels[i]<0?0:(labels[i]>=int(C)?int(C)-1:labels[i]); Lp->grad[i*C+size_t(y)]-=g; } }; return o; }

	inline ValuePtr matmul_add_bias(const ValuePtr&A,const ValuePtr&B,const ValuePtr&bias){
		if(A->shape.size()!=2||B->shape.size()!=2||bias->shape.size()!=1) return nullptr; size_t M=A->shape[0],K=A->shape[1],K2=B->shape[0],N=B->shape[1]; if(K!=K2||bias->shape[0]!=N) return nullptr;
		auto y = matmul(A,B); if(!y) return nullptr;
		for(size_t i=0;i<M;++i){ for(size_t j=0;j<N;++j){ y->values[i*N+j]+=bias->values[j]; }}
		y->parents={A,B,bias}; Value *Ap=A.get(), *Bp=B.get(), *Bpias=bias.get(), *op=y.get();
		y->backward_fn=[op,Ap,Bp,Bpias,M,K,N](){ if(Ap->requires_grad){ for(size_t i=0;i<M;++i){ for(size_t k=0;k<K;++k){ float acc=0.f; for(size_t j=0;j<N;++j) acc+=op->grad[i*N+j]*Bp->values[k*N+j]; Ap->grad[i*K+k]+=acc; }}} if(Bp->requires_grad){ for(size_t k=0;k<K;++k){ for(size_t j=0;j<N;++j){ float acc=0.f; for(size_t i=0;i<M;++i) acc+=Ap->values[i*K+k]*op->grad[i*N+j]; Bp->grad[k*N+j]+=acc; }}} if(Bpias->requires_grad){ for(size_t j=0;j<N;++j){ float acc=0.f; for(size_t i=0;i<M;++i) acc+=op->grad[i*N+j]; Bpias->grad[j]+=acc; }}};
		return y;
	}

	struct Module{ virtual ~Module()=default; virtual ValuePtr forward(const ValuePtr&x)=0; virtual std::vector<ValuePtr> parameters(){ return {}; } virtual void train(bool /*on*/){ } };
	struct Linear:Module{ ValuePtr W,b; size_t in_f=0,out_f=0; explicit Linear(size_t in_,size_t out_,unsigned seed=777):W(randn({in_,out_},seed,0.02f,true)),b(zeros({out_},true)),in_f(in_),out_f(out_){} ValuePtr forward(const ValuePtr&x) override { auto o = matmul_add_bias(x,W,b); if(!o) return add_bias(matmul(x,W),b); return o; } std::vector<ValuePtr> parameters() override { return {W,b}; } };
	struct Sequential:Module{ std::vector<std::shared_ptr<Module>> mods; Sequential()=default; explicit Sequential(std::vector<std::shared_ptr<Module>> m):mods(std::move(m)){} ValuePtr forward(const ValuePtr&x) override { ValuePtr h=x; for(auto &m:mods) h=m->forward(h); return h; } std::vector<ValuePtr> parameters() override { std::vector<ValuePtr> ps; for(auto &m:mods){ auto sub=m->parameters(); ps.insert(ps.end(), sub.begin(), sub.end()); } return ps; } };
	inline std::vector<ValuePtr> collect_parameters(Module&m){ return m.parameters(); }
	struct Optimizer{ virtual ~Optimizer()=default; virtual void step()=0; virtual void zero_grad()=0; virtual void set_lr(float){} virtual float get_lr() const { return 0.0f; } };
	struct SGD:Optimizer{ std::vector<ValuePtr> ps; float lr=1e-2f,wd=0.f; explicit SGD(const std::vector<ValuePtr>&p,float lr_=1e-2f,float wd_=0.f):ps(p),lr(lr_),wd(wd_){} void step() override { for(auto &p:ps){ for(size_t i=0;i<p->numel();++i){ float g=p->grad[i]; if(wd!=0.f) g+=wd*p->values[i]; p->values[i]-=lr*g; } } } void zero_grad() override { for(auto &p:ps) p->zero_grad(); } void set_lr(float v) override { lr=v; } float get_lr() const override { return lr; } };
	struct Adam:Optimizer{ std::vector<ValuePtr> ps; float lr=1e-3f,b1=0.9f,b2=0.999f,eps=1e-8f,wd=0.f; std::unordered_map<Value*,std::vector<float>> m,v; std::unordered_map<Value*,size_t> t; explicit Adam(const std::vector<ValuePtr>&p,float lr_=1e-3f):ps(p),lr(lr_) { for(auto &x:ps){ m[x.get()]=std::vector<float>(x->numel(),0.f); v[x.get()]=std::vector<float>(x->numel(),0.f); t[x.get()]=0; } } void step() override { for(auto &p:ps){ auto &mm=m[p.get()]; auto &vv=v[p.get()]; size_t &tt=t[p.get()]; tt+=1; for(size_t i=0;i<p->numel();++i){ float g=p->grad[i]; if(wd!=0.f) g+=wd*p->values[i]; mm[i]=b1*mm[i]+(1.f-b1)*g; vv[i]=b2*vv[i]+(1.f-b2)*g*g; float mhat=mm[i]/(1.f-std::pow(b1,float(tt))); float vhat=vv[i]/(1.f-std::pow(b2,float(tt))); p->values[i]-=lr*mhat/(std::sqrt(vhat)+eps); } } } void zero_grad() override { for(auto &p:ps) p->zero_grad(); } void set_lr(float v) override { lr=v; } float get_lr() const override { return lr; } };
	struct RMSprop:Optimizer{ std::vector<ValuePtr> ps; float lr=1e-3f, alpha=0.99f, eps=1e-8f, wd=0.f; std::unordered_map<Value*,std::vector<float>> sq; explicit RMSprop(const std::vector<ValuePtr>&p,float lr_=1e-3f):ps(p),lr(lr_) { for(auto &x:ps) sq[x.get()]=std::vector<float>(x->numel(),0.f); } void step() override { for(auto &p:ps){ auto &s=sq[p.get()]; for(size_t i=0;i<p->numel();++i){ float g=p->grad[i]; if(wd!=0.f) g+=wd*p->values[i]; s[i]=alpha*s[i]+(1.f-alpha)*g*g; p->values[i]-=lr*g/(std::sqrt(s[i])+eps); } } } void zero_grad() override { for(auto &p:ps) p->zero_grad(); } };
	struct Adagrad:Optimizer{ std::vector<ValuePtr> ps; float lr=1e-2f, eps=1e-8f, wd=0.f; std::unordered_map<Value*,std::vector<float>> acc; explicit Adagrad(const std::vector<ValuePtr>&p,float lr_=1e-2f):ps(p),lr(lr_) { for(auto &x:ps) acc[x.get()]=std::vector<float>(x->numel(),0.f); } void step() override { for(auto &p:ps){ auto &a=acc[p.get()]; for(size_t i=0;i<p->numel();++i){ float g=p->grad[i]; if(wd!=0.f) g+=wd*p->values[i]; a[i]+=g*g; p->values[i]-=lr*g/(std::sqrt(a[i])+eps); } } } void zero_grad() override { for(auto &p:ps) p->zero_grad(); } };
	struct Adadelta:Optimizer{ std::vector<ValuePtr> ps; float rho=0.95f, eps=1e-6f; std::unordered_map<Value*,std::vector<float>> Eg, Edx; explicit Adadelta(const std::vector<ValuePtr>&p):ps(p){ for(auto &x:ps){ Eg[x.get()]=std::vector<float>(x->numel(),0.f); Edx[x.get()]=std::vector<float>(x->numel(),0.f);} } void step() override { for(auto &p:ps){ auto &eg=Eg[p.get()]; auto &edx=Edx[p.get()]; for(size_t i=0;i<p->numel();++i){ float g=p->grad[i]; eg[i]=rho*eg[i]+(1.f-rho)*g*g; float update = std::sqrt(edx[i]+eps)/std::sqrt(eg[i]+eps)*g; p->values[i]-=update; edx[i]=rho*edx[i]+(1.f-rho)*update*update; } } } void zero_grad() override { for(auto &p:ps) p->zero_grad(); } };

	// Gradient clipping
	inline void clip_grad_norm(const std::vector<ValuePtr>&ps,float max_norm,float eps=1e-6f){ double s=0.0; for(auto &p:ps){ for(float g:p->grad) s += double(g)*double(g); } double n=std::sqrt(s); if(n > double(max_norm)){ float scale = float(double(max_norm)/(n+double(eps))); for(auto &p:ps){ for(float &g:p->grad) g*=scale; } } }
	inline void clip_grad_value(const std::vector<ValuePtr>&ps,float clip){ for(auto &p:ps){ for(float &g:p->grad){ if(g>clip) g=clip; else if(g<-clip) g=-clip; } } }

	// LR schedulers
	struct LRScheduler{ virtual ~LRScheduler()=default; virtual void step()=0; };
	struct StepLR: LRScheduler{ Optimizer &opt; std::size_t step_size; float gamma; std::size_t steps=0; StepLR(Optimizer &o,std::size_t ss,float g):opt(o),step_size(ss),gamma(g){} void step() override { ++steps; if(step_size>0 && (steps % step_size)==0){ opt.set_lr(opt.get_lr()*gamma); } } };
	struct CosineAnnealingLR: LRScheduler{ Optimizer &opt; std::size_t T_max; float eta_min; std::size_t t=0; float base_lr; CosineAnnealingLR(Optimizer&o,std::size_t Tmax,float emin):opt(o),T_max(Tmax),eta_min(emin),t(0),base_lr(o.get_lr()){} void step() override { if(T_max==0){ return; } ++t; float ct = (float)M_PI * (float)std::min<std::size_t>(t, T_max) / float(T_max); float lr = eta_min + (base_lr - eta_min) * 0.5f * (1.0f + std::cos(ct)); opt.set_lr(lr); } };

	// Dropout (inverted)
	inline ValuePtr dropout(const ValuePtr&x,float p,unsigned seed=123){ if(p<=0.f) return x; if(p>=1.f) return zeros(x->shape, x->requires_grad); std::mt19937 rng(seed); std::bernoulli_distribution bern(1.0 - double(p)); float scale = (p>=1.f)? 0.f : (1.0f / (1.0f - p)); auto o=Value::create(x->shape, x->requires_grad); std::vector<float> mask(x->numel()); for(size_t i=0;i<x->numel();++i){ mask[i] = bern(rng) ? scale : 0.0f; o->values[i] = x->values[i] * mask[i]; } o->parents={x}; o->backward_fn=[o,x,mask]() mutable { if(!x->requires_grad) return; for(size_t i=0;i<x->numel();++i) x->grad[i]+=o->grad[i]*mask[i]; }; return o; }
	struct Dropout: Module{ float p; unsigned seed; bool training=true; explicit Dropout(float p_=0.5f,unsigned seed_=123):p(p_),seed(seed_){} ValuePtr forward(const ValuePtr&x) override { if(!training) return x; return dropout(x,p,seed); } void train(bool on) override { training=on; } };

	// LayerNorm (last-dim)
	struct LayerNorm: Module{ std::size_t features; float eps; ValuePtr gamma, beta; explicit LayerNorm(std::size_t features_, float eps_=1e-5f):features(features_),eps(eps_),gamma(ones({features_},true)),beta(zeros({features_},true)){} ValuePtr forward(const ValuePtr&x) override { if(x->shape.size()!=2 || x->shape[1]!=features) return nullptr; size_t B=x->shape[0], C=features; auto o=Value::create(x->shape, x->requires_grad || gamma->requires_grad || beta->requires_grad);
			std::vector<float> mean(B,0.f), var(B,0.f), xhat(B*C,0.f);
			for(size_t i=0;i<B;++i){ double m=0.0; for(size_t c=0;c<C;++c) m+=x->values[i*C+c]; m/=double(C); mean[i]=float(m); double v=0.0; for(size_t c=0;c<C;++c){ double d=double(x->values[i*C+c])-m; v+=d*d; } v/=double(C); var[i]=float(v); float inv_std = 1.0f/std::sqrt(float(v)+eps); for(size_t c=0;c<C;++c){ float nh = (x->values[i*C+c]-mean[i])*inv_std; xhat[i*C+c]=nh; o->values[i*C+c] = nh*gamma->values[c] + beta->values[c]; } }
			o->parents={x,gamma,beta};
			float eps_copy = eps;
			o->backward_fn=[o,x,mean,var,xhat,B,C,gp=gamma,bp=beta,eps_copy](){ if(!(x->requires_grad||gp->requires_grad||bp->requires_grad)) return; std::vector<float> dgamma(C,0.f), dbeta(C,0.f); for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c){ float go=o->grad[i*C+c]; dgamma[c]+= go * xhat[i*C+c]; dbeta[c]+= go; } } if(gp->requires_grad){ for(size_t c=0;c<C;++c) gp->grad[c]+=dgamma[c]; } if(bp->requires_grad){ for(size_t c=0;c<C;++c) bp->grad[c]+=dbeta[c]; } if(x->requires_grad){ for(size_t i=0;i<B;++i){ float inv_std = 1.0f/std::sqrt(var[i]+eps_copy); float sum_dy=0.f, sum_dy_xhat=0.f; for(size_t c=0;c<C;++c){ float go=o->grad[i*C+c]; float dy = go * gp->values[c]; sum_dy += dy; sum_dy_xhat += dy * xhat[i*C+c]; } for(size_t c=0;c<C;++c){ float go=o->grad[i*C+c]; float dy = go * gp->values[c]; float dx = (1.0f/float(C))*inv_std*(float(C)*dy - sum_dy - xhat[i*C+c]*sum_dy_xhat); x->grad[i*C+c] += dx; } } } };
			return o; }
		std::vector<ValuePtr> parameters() override { return {gamma, beta}; }
	};

	// BatchNorm1d over feature dimension [B, C]
	struct BatchNorm1d: Module{ std::size_t features; float eps; float momentum; ValuePtr gamma, beta; std::vector<float> running_mean, running_var; bool training=true; explicit BatchNorm1d(std::size_t features_, float eps_=1e-5f, float momentum_=0.1f):features(features_),eps(eps_),momentum(momentum_),gamma(ones({features_},true)),beta(zeros({features_},true)),running_mean(features_,0.f),running_var(features_,1.f){}
		ValuePtr forward(const ValuePtr&x) override { if(x->shape.size()!=2 || x->shape[1]!=features) return nullptr; size_t B=x->shape[0], C=features; auto o=Value::create(x->shape, x->requires_grad || gamma->requires_grad || beta->requires_grad);
			std::vector<float> mean(C,0.f), var(C,0.f), xhat(B*C,0.f);
			if(training){ for(size_t c=0;c<C;++c){ double m=0.0; for(size_t i=0;i<B;++i) m+=x->values[i*C+c]; m/=double(B); mean[c]=float(m); double v=0.0; for(size_t i=0;i<B;++i){ double d=double(x->values[i*C+c])-m; v+=d*d; } v/=double(B); var[c]=float(v); running_mean[c] = (1.0f - momentum)*running_mean[c] + momentum*mean[c]; running_var[c] = (1.0f - momentum)*running_var[c] + momentum*var[c]; }
			} else { mean = running_mean; var = running_var; }
			for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c){ float inv_std = 1.0f/std::sqrt(var[c]+eps); float nh = (x->values[i*C+c]-mean[c])*inv_std; xhat[i*C+c]=nh; o->values[i*C+c] = nh*gamma->values[c] + beta->values[c]; } }
			o->parents={x,gamma,beta};
			float eps_copy2 = eps;
			o->backward_fn=[o,x,mean,var,xhat,B,C,gp=gamma,bp=beta,eps_copy2](){ if(!(x->requires_grad||gp->requires_grad||bp->requires_grad)) return; std::vector<float> dgamma(C,0.f), dbeta(C,0.f); for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c){ float go=o->grad[i*C+c]; dgamma[c]+= go * xhat[i*C+c]; dbeta[c]+= go; } } if(gp->requires_grad){ for(size_t c=0;c<C;++c) gp->grad[c]+=dgamma[c]; } if(bp->requires_grad){ for(size_t c=0;c<C;++c) bp->grad[c]+=dbeta[c]; } if(x->requires_grad){ std::vector<float> inv_std(C); for(size_t c=0;c<C;++c) inv_std[c]=1.0f/std::sqrt(var[c]+eps_copy2); std::vector<float> sum_dy(C,0.f), sum_dy_xhat(C,0.f); for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c){ float go=o->grad[i*C+c]; float dy = go * gp->values[c]; sum_dy[c]+=dy; sum_dy_xhat[c]+=dy * xhat[i*C+c]; } } for(size_t i=0;i<B;++i){ for(size_t c=0;c<C;++c){ float go=o->grad[i*C+c]; float dy = go * gp->values[c]; float dx = (1.0f/float(B))*inv_std[c]*(float(B)*dy - sum_dy[c] - xhat[i*C+c]*sum_dy_xhat[c]); x->grad[i*C+c] += dx; } } } };
			return o; }
		void train(bool on) override { training=on; }
		std::vector<ValuePtr> parameters() override { return {gamma, beta}; }
	};
	struct TensorDataset{ std::vector<float> X; std::vector<int> y; size_t n=0,d=0; static TensorDataset from(const std::vector<std::vector<float>>&xs,const std::vector<int>&ys){ TensorDataset ds; ds.n=xs.size(); ds.d=xs.empty()?0:xs[0].size(); ds.X.resize(ds.n*ds.d); for(size_t i=0;i<ds.n;++i) std::memcpy(ds.X.data()+static_cast<std::ptrdiff_t>(i*ds.d), xs[i].data(), ds.d*sizeof(float)); ds.y=ys; return ds; } };
	struct DataLoaderConfig{ size_t batch=32; bool shuffle=true; unsigned seed=1234; };
	struct Batch{ ValuePtr x; std::vector<int> y; };
	class DataLoader{ const TensorDataset &ds; DataLoaderConfig cfg; std::vector<size_t> idx; size_t cur=0; public: DataLoader(const TensorDataset&d,DataLoaderConfig c):ds(d),cfg(c){ idx.resize(ds.n); for(size_t i=0;i<ds.n;++i) idx[i]=i; } void reset(){ cur=0; if(cfg.shuffle){ std::mt19937 rng(cfg.seed); std::shuffle(idx.begin(), idx.end(), rng);} } bool next(Batch &out){ if(cur>=idx.size()) return false; size_t b=std::min(cfg.batch, idx.size()-cur); std::vector<float> xb(b*ds.d); std::vector<int> yb(b); for(size_t i=0;i<b;++i){ size_t id=idx[cur+i]; std::memcpy(xb.data()+static_cast<std::ptrdiff_t>(i*ds.d), ds.X.data()+static_cast<std::ptrdiff_t>(id*ds.d), ds.d*sizeof(float)); yb[i]=ds.y[id]; } cur+=b; out.x=tensor(xb,{b,ds.d},false); out.y=std::move(yb); return true; } };
	class Trainer{ public: struct Config{ size_t epochs; Config():epochs(3){} } cfg; explicit Trainer(Config c=Config()):cfg(c){} struct Metrics{ double loss=0.0,acc=0.0; size_t samples=0; }; Metrics fit(Module &model, DataLoader &loader, Optimizer &opt){ loader.reset(); Metrics m{}; size_t correct=0; Batch batch{}; for(size_t e=0;e<cfg.epochs;++e){ loader.reset(); while(loader.next(batch)){ auto logits=model.forward(batch.x); auto loss=cross_entropy_logits(logits, batch.y); opt.zero_grad(); loss->backward(); opt.step(); m.loss += loss->values[0]*double(batch.y.size()); m.samples += batch.y.size(); size_t B=logits->shape[0], C=logits->shape[1]; for(size_t i=0;i<B;++i){ size_t arg=0; float best=logits->values[i*C+0]; for(size_t c=1;c<C;++c){ float v=logits->values[i*C+c]; if(v>best){ best=v; arg=c; } } if(int(arg)==batch.y[i]) ++correct; } } } m.acc = (m.samples==0)?0.0:double(correct)/double(m.samples); return m; } };

	// Convenience: printable summary
	inline std::string summary_str(Module&m){ auto ps=collect_parameters(m); size_t total=0; for(auto &p:ps) total+=p->numel(); std::ostringstream oss; oss<<"Parameters: "<<ps.size()<<", scalars: "<<total; return oss.str(); }

	// Error-handling wrappers with informative messages
	inline StatusOr<ValuePtr> try_add(const ValuePtr&a,const ValuePtr&b){ if(!a||!b) return make_status(StatusCode::kInvalidArgument, "add: null operand"); if(!same(a->shape,b->shape)) { std::ostringstream oss; oss<<"add: shape mismatch "; oss<<"["; for(size_t i=0;i<a->shape.size();++i){ oss<<a->shape[i]<<(i+1<a->shape.size()?"x":""); } oss<<"] vs ["; for(size_t i=0;i<b->shape.size();++i){ oss<<b->shape[i]<<(i+1<b->shape.size()?"x":""); } oss<<"]"; return make_status(StatusCode::kInvalidArgument, make_status_message("add: shape mismatch", oss.str().c_str())); } return add(a,b); }
	inline StatusOr<ValuePtr> try_mul(const ValuePtr&a,const ValuePtr&b){ if(!a||!b) return make_status(StatusCode::kInvalidArgument, "mul: null operand"); if(!same(a->shape,b->shape)) { std::ostringstream oss; oss<<"mul: shape mismatch "; oss<<"["; for(size_t i=0;i<a->shape.size();++i){ oss<<a->shape[i]<<(i+1<a->shape.size()?"x":""); } oss<<"] vs ["; for(size_t i=0;i<b->shape.size();++i){ oss<<b->shape[i]<<(i+1<b->shape.size()?"x":""); } oss<<"]"; return make_status(StatusCode::kInvalidArgument, make_status_message("mul: shape mismatch", oss.str().c_str())); } return mul(a,b); }
	inline StatusOr<ValuePtr> try_matmul(const ValuePtr&A,const ValuePtr&B){ if(!A||!B) return make_status(StatusCode::kInvalidArgument, "matmul: null operand"); if(A->shape.size()!=2||B->shape.size()!=2) return make_status(StatusCode::kInvalidArgument, "matmul: rank must be 2"); if(A->shape[1]!=B->shape[0]){ std::ostringstream oss; oss<<"matmul: inner dims mismatch ("<<A->shape[1]<<" vs "<<B->shape[0]<<")"; return make_status(StatusCode::kInvalidArgument, make_status_message("matmul dims", oss.str().c_str())); } auto out=matmul(A,B); if(!out) return make_status(StatusCode::kInternal, "matmul failed"); return out; }
	inline StatusOr<ValuePtr> try_add_bias(const ValuePtr&x,const ValuePtr&b){ if(!x||!b) return make_status(StatusCode::kInvalidArgument, "add_bias: null operand"); if(x->shape.size()!=2||b->shape.size()!=1||x->shape[1]!=b->shape[0]) return make_status(StatusCode::kInvalidArgument, "add_bias: shape mismatch (expected [B,N]+[N])"); auto out=add_bias(x,b); if(!out) return make_status(StatusCode::kInternal, "add_bias failed"); return out; }

	// Simple training helper
	struct SimpleFitConfig{ size_t epochs=3; size_t batch=64; float lr=1e-3f; bool use_adam=true; unsigned seed=1234; };
	struct SimpleFitMetrics{ double avg_loss=0.0; double acc=0.0; size_t samples=0; };
	inline StatusOr<SimpleFitMetrics> fit_simple(Module &model, const TensorDataset &ds, SimpleFitConfig cfg = {}){
		DataLoader loader(ds, DataLoaderConfig{cfg.batch, true, cfg.seed});
		auto params = collect_parameters(model);
		std::unique_ptr<Optimizer> opt;
		if(cfg.use_adam) opt.reset(new Adam(params, cfg.lr)); else opt.reset(new SGD(params, cfg.lr));
		Trainer::Config tcfg; tcfg.epochs = cfg.epochs; Trainer trainer(tcfg);
		auto m = trainer.fit(model, loader, *opt);
		SimpleFitMetrics r; r.avg_loss = (m.samples==0?0.0:m.loss/double(m.samples)); r.acc = m.acc; r.samples = m.samples; return r;
	}

	// Regression dataset/loader and trainer
	struct TensorDatasetReg{ std::vector<float> X; std::vector<float> y; size_t n=0,d=0; static TensorDatasetReg from(const std::vector<std::vector<float>>&xs,const std::vector<float>&ys){ TensorDatasetReg ds; ds.n=xs.size(); ds.d=xs.empty()?0:xs[0].size(); ds.X.resize(ds.n*ds.d); for(size_t i=0;i<ds.n;++i) std::memcpy(ds.X.data()+static_cast<std::ptrdiff_t>(i*ds.d), xs[i].data(), ds.d*sizeof(float)); ds.y=ys; return ds; } };
	struct BatchReg{ ValuePtr x; ValuePtr y; };
	class DataLoaderReg{ const TensorDatasetReg &ds; DataLoaderConfig cfg; std::vector<size_t> idx; size_t cur=0; public: DataLoaderReg(const TensorDatasetReg&d,DataLoaderConfig c):ds(d),cfg(c){ idx.resize(ds.n); for(size_t i=0;i<ds.n;++i) idx[i]=i; } void reset(){ cur=0; if(cfg.shuffle){ std::mt19937 rng(cfg.seed); std::shuffle(idx.begin(), idx.end(), rng);} } bool next(BatchReg &out){ if(cur>=idx.size()) return false; size_t b=std::min(cfg.batch, idx.size()-cur); std::vector<float> xb(b*ds.d); std::vector<float> yb(b); for(size_t i=0;i<b;++i){ size_t id=idx[cur+i]; std::memcpy(xb.data()+static_cast<std::ptrdiff_t>(i*ds.d), ds.X.data()+static_cast<std::ptrdiff_t>(id*ds.d), ds.d*sizeof(float)); yb[i]=ds.y[id]; } cur+=b; out.x=tensor(xb,{b,ds.d},false); out.y=tensor(yb,{b,1},false); return true; } };
	inline Trainer::Metrics fit_regression(Module &model, DataLoaderReg &loader, Optimizer &opt, size_t epochs=3){ loader.reset(); Trainer::Metrics m{}; BatchReg batch{}; for(size_t e=0;e<epochs;++e){ loader.reset(); while(loader.next(batch)){ auto ypred = model.forward(batch.x); auto loss = mse_loss(ypred, batch.y); opt.zero_grad(); loss->backward(); opt.step(); m.loss += loss->values[0]*double(ypred->shape[0]); m.samples += ypred->shape[0]; } } return m; }

	// Checkpoint save/load for modules
	inline Status save_module(Module &m, const std::string &path){ auto ps=collect_parameters(m); FILE *f = std::fopen(path.c_str(), "wb"); if(!f) return make_status_errno(StatusCode::kFailedPrecondition, "save_module: fopen failed", errno); uint64_t np = static_cast<uint64_t>(ps.size()); if(std::fwrite(&np, sizeof(uint64_t), 1, f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "save_module: write np failed"); } for(auto &p:ps){ uint64_t nd=static_cast<uint64_t>(p->shape.size()); if(std::fwrite(&nd,sizeof(uint64_t),1,f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "save_module: write nd failed"); } for(size_t i=0;i<p->shape.size();++i){ uint64_t d=static_cast<uint64_t>(p->shape[i]); if(std::fwrite(&d,sizeof(uint64_t),1,f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "save_module: write dim failed"); } } uint64_t n=static_cast<uint64_t>(p->numel()); if(std::fwrite(&n,sizeof(uint64_t),1,f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "save_module: write n failed"); } if(n>0 && std::fwrite(p->values.data(), sizeof(float), static_cast<size_t>(n), f)!=n){ std::fclose(f); return make_status(StatusCode::kInternal, "save_module: write values failed"); } } std::fclose(f); return Status::OK(); }
	inline Status load_module(Module &m, const std::string &path){ auto ps=collect_parameters(m); FILE *f = std::fopen(path.c_str(), "rb"); if(!f) return make_status_errno(StatusCode::kFailedPrecondition, "load_module: fopen failed", errno); uint64_t np=0; if(std::fread(&np,sizeof(uint64_t),1,f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "load_module: read np failed"); } if(np!=ps.size()){ std::fclose(f); return make_status(StatusCode::kFailedPrecondition, "load_module: param count mismatch"); } for(size_t pi=0; pi<ps.size(); ++pi){ uint64_t nd=0; if(std::fread(&nd,sizeof(uint64_t),1,f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "load_module: read nd failed"); } std::vector<std::size_t> dims(nd); for(size_t i=0;i<nd;++i){ uint64_t d=0; if(std::fread(&d,sizeof(uint64_t),1,f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "load_module: read dim failed"); } dims[i]=static_cast<std::size_t>(d); } uint64_t n=0; if(std::fread(&n,sizeof(uint64_t),1,f)!=1){ std::fclose(f); return make_status(StatusCode::kInternal, "load_module: read n failed"); } if(dims!=ps[pi]->shape || n!=ps[pi]->numel()){ std::fclose(f); return make_status(StatusCode::kFailedPrecondition, "load_module: shape/numel mismatch"); } if(n>0 && std::fread(ps[pi]->values.data(), sizeof(float), static_cast<size_t>(n), f)!=n){ std::fclose(f); return make_status(StatusCode::kInternal, "load_module: read values failed"); } } std::fclose(f); return Status::OK(); }
	inline void summary(Module&m,std::ostream &os=std::cout){ auto ps=collect_parameters(m); size_t total=0; for(auto &p:ps) total+=p->numel(); os<<"Parameters: "<<ps.size()<<", scalars: "<<total<<"\n"; }
} // namespace nn

// Extended reward functions
inline float reward_f1_binary(const std::vector<int>&pred,const std::vector<int>&lab){ size_t tp=0,fp=0,fn=0; size_t n=std::min(pred.size(),lab.size()); for(size_t i=0;i<n;++i){ int p=pred[i]?1:0; int y=lab[i]?1:0; if(p==1&&y==1)++tp; else if(p==1&&y==0)++fp; else if(p==0&&y==1)++fn; } float pr=(tp+fp==0)?0.f:float(tp)/float(tp+fp); float rc=(tp+fn==0)?0.f:float(tp)/float(tp+fn); return (pr+rc==0.f)?0.f:(2.f*pr*rc/(pr+rc)); }
inline float reward_bleu_1_4(const std::vector<int>&pred,const std::vector<int>&ref,size_t max_n=4){
    if(pred.empty()||ref.empty()) return 0.f;
    max_n=std::max<size_t>(1,std::min<size_t>(max_n,4));
    auto counts=[&](const std::vector<int>&s,size_t n){
        std::unordered_map<std::string,size_t> m;
        if(s.size()<n) return m;
        for(size_t i=0;i+n<=s.size();++i){
            std::string k; k.reserve(n*4);
            for(size_t j=0;j<n;++j){ k.append(std::to_string(s[i+j])); k.push_back(','); }
            ++m[k];
        }
        return m;
    };
    double logp=0.0;
    for(size_t n=1;n<=max_n;++n){
        auto cp=counts(pred,n), cr=counts(ref,n);
        int match=0, tot=0;
        for(auto &kv:cp){ int p=static_cast<int>(kv.second); int r=static_cast<int>(cr[kv.first]); match+=std::min(p,r); tot+=p; }
        double pn=(tot==0)?0.0:double(match)/double(tot);
        logp += (pn<=0.0)?-1e9:std::log(pn);
    }
    double bp=1.0;
    if(pred.size()<ref.size()) bp=std::exp(1.0-double(ref.size())/double(pred.size()));
    return float(bp*std::exp(logp/double(max_n)));
}

inline float reward_rouge_l(const std::vector<int>&pred,const std::vector<int>&ref){
    size_t n=pred.size(), m=ref.size(); if(n==0||m==0) return 0.f;
    std::vector<size_t> dp(m+1,0);
    for(size_t i=1;i<=n;++i){ size_t prev=0; for(size_t j=1;j<=m;++j){ size_t tmp=dp[j]; if(pred[i-1]==ref[j-1]) dp[j]=prev+1; else dp[j]=std::max(dp[j], dp[j-1]); prev=tmp; } }
    float lcs=float(dp[m]); float pr=lcs/float(n), rc=lcs/float(m); return (pr+rc==0.f)?0.f:(2.f*pr*rc/(pr+rc));
}

// Additional evaluation metrics
inline void metrics_precision_recall_f1_binary(const std::vector<int>&pred,const std::vector<int>&lab,float &precision,float &recall,float &f1){
	size_t tp=0,fp=0,fn=0; size_t n=std::min(pred.size(),lab.size());
	for(size_t i=0;i<n;++i){ int p=pred[i]?1:0; int y=lab[i]?1:0; if(p==1&&y==1)++tp; else if(p==1&&y==0)++fp; else if(p==0&&y==1)++fn; }
	precision = (tp+fp==0)?0.f:float(tp)/float(tp+fp);
	recall = (tp+fn==0)?0.f:float(tp)/float(tp+fn);
	f1 = (precision+recall==0.f)?0.f:(2.f*precision*recall/(precision+recall));
}

inline float auc_roc_binary(const std::vector<float>&scores,const std::vector<int>&labels){
	const size_t n = std::min(scores.size(), labels.size());
	if(n==0) return 0.f; std::vector<size_t> idx(n); for(size_t i=0;i<n;++i) idx[i]=i; std::sort(idx.begin(), idx.end(), [&](size_t a,size_t b){ return scores[a] > scores[b]; });
	// Count positives and negatives
	double pos=0.0, neg=0.0; for(size_t i=0;i<n;++i){ if(labels[i]) pos+=1.0; else neg+=1.0; }
	if(pos==0.0 || neg==0.0) return 0.f;
	// Compute AUC by rank-sum (Wilcoxon)
	double rank_sum=0.0; for(size_t r=0;r<n;++r){ if(labels[idx[r]]) rank_sum += double(r+1); }
	double auc = (rank_sum - pos*(pos+1.0)/2.0) / (pos*neg);
	if(auc<0.0) auc=0.0; if(auc>1.0) auc=1.0; return float(auc);
}

// ===== onnx (stub) =====
inline Status import_onnx_model(const std::string &path) {
    (void)path;
    if (global_config().deterministic) {
        // In deterministic mode, skip random initializations, etc. Stub ok.
        return Status::OK();
    }
    return Status::OK();
}

// GPU FWHT removed; CPU-only build

} // namespace kllm