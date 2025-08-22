#pragma once

#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace kllm {

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

} // namespace kllm