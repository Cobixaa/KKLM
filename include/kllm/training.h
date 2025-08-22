#pragma once

#include <vector>
#include <cstddef>
#include <cmath>

#include "kllm/reward.h"
#include "kllm/quant.h"
#include "kllm/status.h"

namespace kllm {

struct OptimizerConfig {
	float lr = 1e-2f;
};

// Simple block-wise SGD on a vector of parameters with gradient
inline void sgd_step(std::vector<float> &params, const std::vector<float> &grads, const OptimizerConfig &cfg) {
	const std::size_t n = params.size();
	for (std::size_t i = 0; i < n; ++i) {
		params[i] -= cfg.lr * grads[i];
	}
}

// Approximate backprop placeholder: gradient = prediction - target
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

} // namespace kllm