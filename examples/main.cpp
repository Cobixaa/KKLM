#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "kllm/kllm.h"

static void test_fwht() {
	std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	auto st = kllm::fwht(x);
	if (!st.ok()) std::cout << "FWHT error: " << st.message << "\n";
	st = kllm::fwht_inverse(x);
	if (!st.ok()) std::cout << "FWHT inv error: " << st.message << "\n";
	float max_abs_err = 0.0f;
	const float expected[] = {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f};
	for (std::size_t i = 0; i < x.size(); ++i) {
		max_abs_err = std::max(max_abs_err, std::fabs(x[i] - expected[i]));
	}
	std::cout << "FWHT invertibility max abs err: " << max_abs_err << "\n";
}

static void test_ntt() {
	std::vector<std::uint32_t> a = {1,2,3,4,5,6,7,8};
	std::vector<std::uint32_t> b = a;
	bool ok = kllm::ntt_inplace(b, false);
	bool ok2 = kllm::ntt_inplace(b, true);
	(void)ok;
	(void)ok2;
	std::size_t mismatches = 0;
	for (std::size_t i = 0; i < a.size(); ++i) {
		if (b[i] != a[i]) ++mismatches;
	}
	std::cout << "NTT invertibility mismatches: " << mismatches << "\n";
}

static void test_countsketch() {
	std::vector<float> x(16);
	for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i + 1);
	kllm::CountSketch cs(8, 3);
	std::vector<float> y(8, 0.0f);
	cs.apply(x.data(), x.size(), y.data());
	float checksum = 0.0f;
	for (float v : y) checksum += v;
	std::cout << "CountSketch checksum: " << checksum << "\n";
}

static void test_fused() {
	std::vector<float> src(8);
	for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i);
	std::vector<float> dst(8, 0.0f);
	kllm::fused_fwht_scale_add(src.data(), src.size(), 0.5f, dst.data());
	float checksum = 0.0f;
	for (float v : dst) checksum += v;
	std::cout << "Fused FWHT-scale-add checksum: " << checksum << "\n";
}

static void test_simple_bias_relu() {
	std::vector<float> x(8);
	std::vector<float> b(8, 0.5f);
	for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i % 2 == 0) ? -1.5f : 1.0f;
	std::vector<float> y;
	auto st = kllm::fused_transform_bias_relu(x, b, y);
	if (!st.ok()) std::cout << "Fused API error: " << st.message << "\n";
	float sum = 0.0f;
	for (float v : y) sum += v;
	std::cout << "Fused API FWHT+bias+ReLU sum: " << sum << "\n";
}

static void test_quant() {
	std::vector<float> x(16);
	for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i - 8) * 0.25f;
	kllm::QuantParams qp{};
	std::vector<int8_t> q;
	std::vector<float> deq;
	auto st = kllm::quantize_dequantize(x, q, deq, qp);
	if (!st.ok()) std::cout << "Quant API error: " << st.message << "\n";
	float max_abs_err = 0.0f;
	for (std::size_t i = 0; i < x.size(); ++i) max_abs_err = std::max(max_abs_err, std::fabs(x[i] - deq[i]));
	std::cout << "Quant max abs error: " << max_abs_err << " (scale=" << qp.scale << ")\n";
}

static void test_rewards() {
	std::vector<float> pred = {0.1f, 0.9f, -0.2f, 1.0f};
	std::vector<float> target = {0.0f, 1.0f, 0.0f, 1.0f};
	float mse = kllm::reward_mse(pred, target);
	float cos = kllm::reward_cosine_similarity(pred, target);
	std::vector<float> logits = {0.2f, 0.8f, 0.6f, 0.4f};
	std::vector<int> labels = {1, 0};
	float acc = kllm::reward_top1_accuracy(logits, labels, 2);
	std::cout << "Reward MSE=" << mse << ", cosine=" << cos << ", acc=" << acc << "\n";
}

static void test_ir() {
	std::vector<float> x(8);
	for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i % 2 == 0) ? -1.0f : 2.0f;
	auto n0 = kllm::GraphBuilder::input(x);
	auto n1 = kllm::GraphBuilder::transform(n0);
	auto n2 = kllm::GraphBuilder::relu(n1);
	auto planned = kllm::Planner::plan(n2);
	kllm::Tensor out = planned->evaluate();
	float sum = 0.0f;
	for (float v : out.values) sum += v;
	std::cout << "IR planned Transform+Relu sum: " << sum << "\n";
}

int main() {
	test_fwht();
	test_ntt();
	test_countsketch();
	test_fused();
	test_simple_bias_relu();
	test_quant();
	test_rewards();
	test_ir();
	return 0;
}