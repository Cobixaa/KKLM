#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "kllm/kllm.h"

static void test_fwht() {
	std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
	kllm::fwht_inplace(x.data(), x.size());
	kllm::fwht_inplace_inverse(x.data(), x.size());
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

static void test_fused_bias_relu() {
	std::vector<float> x(8);
	std::vector<float> b(8, 0.5f);
	for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i % 2 == 0) ? -1.5f : 1.0f;
	std::vector<float> y(8, 0.0f);
	kllm::fused_fwht_bias_relu(x.data(), b.data(), x.size(), y.data());
	float sum = 0.0f;
	for (float v : y) sum += v;
	std::cout << "Fused FWHT+bias+ReLU sum: " << sum << "\n";
}

static void test_quant() {
	std::vector<float> x(16);
	for (std::size_t i = 0; i < x.size(); ++i) x[i] = (i - 8) * 0.25f;
	auto qp = kllm::choose_symmetric_int8_scale(x.data(), x.size());
	std::vector<int8_t> q(x.size());
	kllm::quantize_int8(x.data(), x.size(), q.data(), qp);
	std::vector<float> deq(x.size());
	kllm::dequantize_int8(q.data(), q.size(), qp.scale, deq.data());
	float max_abs_err = 0.0f;
	for (std::size_t i = 0; i < x.size(); ++i) max_abs_err = std::max(max_abs_err, std::fabs(x[i] - deq[i]));
	std::cout << "Quant max abs error: " << max_abs_err << " (scale=" << qp.scale << ")\n";
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
	test_fused_bias_relu();
	test_quant();
	test_ir();
	return 0;
}