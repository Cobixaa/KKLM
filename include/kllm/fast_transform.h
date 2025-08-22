#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__AVX2__)
	#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	#include <arm_neon.h>
#endif

#include "kllm/utils.h"
#include "kllm/parallel.h"

namespace kllm {

// In-place Fast Walsh-Hadamard Transform for contiguous float data.
// Self-inverse up to scaling by N. Caller ensures length is a power of two.
inline void fwht_inplace(float *data, std::size_t length) {
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
				__m256 a = _mm256_loadu_ps(a_ptr);
				__m256 b = _mm256_loadu_ps(b_ptr);
				__m256 sum = _mm256_add_ps(a, b);
				__m256 diff = _mm256_sub_ps(a, b);
				_mm256_storeu_ps(a_ptr, sum);
				_mm256_storeu_ps(b_ptr, diff);
			}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
			// NEON path (4 floats per vector)
			const std::size_t neon_width = 4;
			for (; i + neon_width <= half_block; i += neon_width) {
				float *a_ptr = data + block_start + i;
				float *b_ptr = data + block_start + i + half_block;
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

inline void fwht_inplace_parallel(float *data, std::size_t length, ThreadPool &pool) {
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

// Scales data by 1/length to invert the FWHT (FWHT is its own inverse up to N)
inline void fwht_inplace_inverse(float *data, std::size_t length) {
	if (data == nullptr || !is_power_of_two(length)) {
		return;
	}
	fwht_inplace(data, length);
	const float scale = 1.0f / static_cast<float>(length);
	for (std::size_t i = 0; i < length; ++i) {
		data[i] *= scale;
	}
}

} // namespace kllm