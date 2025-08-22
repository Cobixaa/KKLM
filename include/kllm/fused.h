#pragma once

#include <cstddef>

#include "kllm/fast_transform.h"

namespace kllm {

inline void fused_fwht_scale_add(const float *input, std::size_t length, float scale, float *inout_destination) {
	if (input == nullptr || inout_destination == nullptr || !is_power_of_two(length)) {
		return;
	}
	// Copy input into a temporary buffer to avoid mutating it
	std::vector<float> buffer(length);
	for (std::size_t i = 0; i < length; ++i) {
		buffer[i] = input[i];
	}
	fwht_inplace(buffer.data(), length);
	for (std::size_t i = 0; i < length; ++i) {
		inout_destination[i] += scale * buffer[i];
	}
}

} // namespace kllm