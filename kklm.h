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

// ===== config =====
struct Config {
	bool deterministic = false;
	std::size_t num_threads = 0; // 0 => use hardware_concurrency
	std::size_t prefetch_distance = 32; // elements ahead for prefetching (floats)
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

template <typename T>
inline std::unique_ptr<T, FreeDeleter> allocate_aligned(std::size_t count, std::size_t alignment) {
	const std::size_t total = count * sizeof(T);
	auto raw = allocate_aligned_bytes(total, alignment);
	return std::unique_ptr<T, FreeDeleter>(static_cast<T *>(raw.release()));
}

inline bool set_current_thread_affinity(int cpu_index) {
	(void)cpu_index;
#if defined(__linux__)
	if (cpu_index < 0) {
		return false;
	}
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(static_cast<unsigned int>(cpu_index), &set);
	const int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
	return result == 0;
#else
	return false;
#endif
}

// ===== parallel =====
class ThreadPool {
public:
	explicit ThreadPool(std::size_t num_threads) : stop_flag(false), outstanding_tasks(0) {
		if (num_threads == 0) num_threads = 1;
		workers.reserve(num_threads);
		for (std::size_t i = 0; i < num_threads; ++i) {
			workers.emplace_back([this]() { this->worker_loop(); });
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
			task();
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
KLLM_INLINE void fwht_inplace(float * KLLM_RESTRICT data, std::size_t length) {
	if (data == nullptr || !is_power_of_two(length)) {
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
				__m256 a;
				__m256 b;
				if (is_aligned_32(a_ptr) && is_aligned_32(b_ptr)) {
					a = _mm256_load_ps(a_ptr);
					b = _mm256_load_ps(b_ptr);
				} else {
					a = _mm256_loadu_ps(a_ptr);
					b = _mm256_loadu_ps(b_ptr);
				}
				__m256 sum = _mm256_add_ps(a, b);
				__m256 diff = _mm256_sub_ps(a, b);
				if (is_aligned_32(a_ptr) && is_aligned_32(b_ptr)) {
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
			const std::size_t vec_width = 8;
			for (; i + vec_width <= half_block; i += vec_width) {
				float *a_ptr = data + block_start + i;
				float *b_ptr = data + block_start + i + half_block;
				prefetch(a_ptr + global_config().prefetch_distance);
				prefetch(b_ptr + global_config().prefetch_distance);
				__m256 a = _mm256_loadu_ps(a_ptr);
				__m256 b = _mm256_loadu_ps(b_ptr);
				__m256 sum = _mm256_add_ps(a, b);
				__m256 diff = _mm256_sub_ps(a, b);
				_mm256_storeu_ps(a_ptr, sum);
				_mm256_storeu_ps(b_ptr, diff);
			}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
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
	fwht_inplace(buffer.data(), length);
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
}

KLLM_INLINE void fused_fwht_bias_relu(const float * KLLM_RESTRICT input, const float * KLLM_RESTRICT bias, std::size_t length, float * KLLM_RESTRICT destination) {
	if (input == nullptr || bias == nullptr || destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	static thread_local std::vector<float> buffer;
	buffer.resize(length);
	std::memcpy(buffer.data(), input, length * sizeof(float));
	fwht_inplace(buffer.data(), length);
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
}

KLLM_INLINE void fused_fwht_bias_gelu(const float * KLLM_RESTRICT input, const float * KLLM_RESTRICT bias, std::size_t length, float * KLLM_RESTRICT destination) {
	if (input == nullptr || bias == nullptr || destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	static thread_local std::vector<float> buffer;
	buffer.resize(length);
	std::memcpy(buffer.data(), input, length * sizeof(float));
	fwht_inplace(buffer.data(), length);
	for (std::size_t i = 0; i < length; ++i) {
		float x = buffer[i] + bias[i];
		float t = x * (0.7978845608028654f) * (1.0f + 0.044715f * x * x); // sqrt(2/pi)
		float y = 0.5f * x * (1.0f + std::tanh(t));
		destination[i] = y;
	}
}

KLLM_INLINE void fused_fwht_bias_silu(const float * KLLM_RESTRICT input, const float * KLLM_RESTRICT bias, std::size_t length, float * KLLM_RESTRICT destination) {
	if (input == nullptr || bias == nullptr || destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	static thread_local std::vector<float> buffer;
	buffer.resize(length);
	std::memcpy(buffer.data(), input, length * sizeof(float));
	fwht_inplace(buffer.data(), length);
	for (std::size_t i = 0; i < length; ++i) {
		float x = buffer[i] + bias[i];
		float s = 1.0f / (1.0f + std::exp(-x));
		destination[i] = x * s; // swish
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
		for (std::size_t i = 0; i < n; i += len) {
			std::uint32_t w = 1u;
			for (std::size_t j = 0; j < half; ++j) {
				const std::uint32_t u = a[i + j];
				const std::uint32_t v = mul_mod(a[i + j + half], w);
				a[i + j] = add_mod(u, v);
				a[i + j + half] = sub_mod(u, v);
				w = mul_mod(w, wlen);
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
	const std::size_t n = pred.size();
	grad_out.resize(n);
	for (std::size_t i = 0; i < n; ++i) {
		grad_out[i] = pred[i] - target[i];
	}
}

struct TrainingStepResult {
	float loss;
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
	// d loss / d params[0] = sum(grad_pred[i] * inputs[i])
	float dparam0 = 0.0f;
	for (std::size_t i = 0; i < inputs.size(); ++i) dparam0 += grad_pred[i] * inputs[i];
	std::vector<float> grads(params.size(), 0.0f);
	if (!params.empty()) grads[0] = dparam0;
	sgd_step(params, grads, opt);
	out.loss = loss;
	return Status::OK();
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

} // namespace kllm