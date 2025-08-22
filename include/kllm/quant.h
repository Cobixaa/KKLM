#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>

#if defined(__AVX2__)
	#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
	#include <arm_neon.h>
#endif

namespace kllm {

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
		alignas(32) int32_t tmp[8];
		_mm256_store_si256(reinterpret_cast<__m256i *>(tmp), y_i32);
		for (int k = 0; k < 8; ++k) {
			int v = tmp[k];
			if (v > 127) v = 127;
			if (v < -127) v = -127;
			output[i + static_cast<std::size_t>(k)] = static_cast<int8_t>(v);
		}
	}
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
	// Use scalar fallback for simplicity and portability
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
	for (std::size_t i = 0; i < length; ++i) {
		output[i] = static_cast<float>(input[i]) * s;
	}
}

} // namespace kllm