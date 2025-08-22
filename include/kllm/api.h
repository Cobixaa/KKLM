#pragma once

#include <cstddef>
#include <vector>

#include "kllm/fast_transform.h"
#include "kllm/fused.h"
#include "kllm/quant.h"
#include "kllm/status.h"

namespace kllm {

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

} // namespace kllm