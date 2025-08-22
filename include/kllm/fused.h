#pragma once

#include <cstddef>
#include <vector>

#include "kllm/fast_transform.h"
#include "kllm/utils.h"

#if defined(__AVX2__)
	#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	#include <arm_neon.h>
#endif

namespace kllm {

inline void fused_fwht_scale_add(const float *input, std::size_t length, float scale, float *inout_destination) {
	if (input == nullptr || inout_destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	std::vector<float> buffer(length);
	for (std::size_t i = 0; i < length; ++i) {
		buffer[i] = input[i];
	}
	fwht_inplace(buffer.data(), length);
	std::size_t i = 0;
#if defined(__AVX2__)
	const __m256 vscale = _mm256_set1_ps(scale);
	for (; i + 8 <= length; i += 8) {
		__m256 u = _mm256_loadu_ps(buffer.data() + i);
		__m256 acc = _mm256_loadu_ps(inout_destination + i);
		acc = _mm256_fmadd_ps(u, vscale, acc);
		_mm256_storeu_ps(inout_destination + i, acc);
	}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
	float32x4_t vscale = vdupq_n_f32(scale);
	for (; i + 4 <= length; i += 4) {
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

inline void fused_fwht_bias_relu(const float *input, const float *bias, std::size_t length, float *destination) {
	if (input == nullptr || bias == nullptr || destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	std::vector<float> buffer(length);
	for (std::size_t i = 0; i < length; ++i) buffer[i] = input[i];
	fwht_inplace(buffer.data(), length);
	std::size_t i = 0;
#if defined(__AVX2__)
	for (; i + 8 <= length; i += 8) {
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

} // namespace kllm