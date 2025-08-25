#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "kklm.h"

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

static void test_fused_extra() {
	std::vector<float> src(8);
	for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i) - 4.0f;
	std::vector<float> bias(8, 0.1f);
	std::vector<float> out1(8), out2(8);
	kllm::fused_fwht_bias_gelu(src.data(), bias.data(), src.size(), out1.data());
	kllm::fused_fwht_bias_silu(src.data(), bias.data(), src.size(), out2.data());
	float n1 = kllm::l2_norm(out1.data(), out1.size());
	float n2 = kllm::l2_norm(out2.data(), out2.size());
	std::cout << "Fused GELU/SiLU norms: " << n1 << ", " << n2 << "\n";
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
	kllm::Profiler p;
	kllm::Scheduler sched(p);
	auto plan = sched.plan(n2);
	kllm::Tensor out = plan.root->evaluate();
	float sum = 0.0f;
	for (float v : out.values) sum += v;
	std::cout << "IR+Scheduler sum: " << sum << ", fused=" << (plan.used_fusion ? 1 : 0) << "\n";
}

static void test_transforms_extras() {
	// Low-rank sanity
	const std::size_t in_dim = 4, out_dim = 3, rank = 2;
	std::vector<float> U = {
		1, 0,
		0, 1,
		1, 1
	};
	std::vector<float> V = {
		1, 0,
		0, 1,
		1, 0,
		0, 1
	};
	std::vector<float> x = {1,2,3,4};
	std::vector<float> y(out_dim);
	kllm::low_rank_apply(U.data(), V.data(), out_dim, in_dim, rank, x.data(), y.data());
	float sum = 0.0f; for (float v : y) sum += v;
	std::cout << "Low-rank sum: " << sum << "\n";
}

static void test_config_and_onnx() {
	kllm::set_deterministic(true);
	auto st = kllm::import_onnx_model("dummy.onnx");
	if (!st.ok()) std::cout << "ONNX import error: " << st.message << "\n";
}

static void test_training() {
	std::vector<float> params = {1.0f};
	std::vector<float> inputs = {1,2,3,4};
	std::vector<float> targets = {1.1f, 1.9f, 3.1f, 3.9f};
	kllm::TrainingStepResult res{};
	kllm::OptimizerConfig opt; opt.lr = 0.1f;
	auto st = kllm::train_step_mse(params, inputs, targets, opt, res);
	if (!st.ok()) std::cout << "Train error: " << st.message << "\n";
	std::cout << "Training step loss: " << res.loss << ", param0: " << params[0] << "\n";
}

static void test_nn_regression() {
	using namespace kllm::nn;
	// Simple linear regression y = x0 + x1 with noise
	std::vector<std::vector<float>> X(100, std::vector<float>(2));
	std::vector<float> Y(100);
	for (size_t i = 0; i < 100; ++i) { X[i][0] = float(i)*0.01f; X[i][1] = 1.0f; Y[i] = X[i][0] + 0.5f; }
	auto ds = kllm::nn::TensorDatasetReg::from(X, Y);
	kllm::nn::DataLoaderReg loader(ds, kllm::nn::DataLoaderConfig{32, true, 123});
	kllm::nn::Sequential net({ std::make_shared<kllm::nn::Linear>(2, 1) });
	auto params = kllm::nn::collect_parameters(net);
	kllm::nn::Adam opt(params, 1e-2f);
	auto metrics = kllm::nn::fit_regression(net, loader, opt, 3);
	std::cout << "nn regression avg loss=" << (metrics.samples==0?0.0:metrics.loss/double(metrics.samples)) << "\n";
}

static void test_pipeline_v21() {
	std::vector<float> x(1 << 16);
	std::mt19937 rng(123);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (float &v : x) v = dist(rng);
	kllm::PipelineTelemetry t{};
	std::vector<int8_t> q8; std::vector<float> sc8; std::vector<uint8_t> q4; std::vector<float> sc4;
	auto st1 = kllm::run_pipeline_v21_to_int8(x, 1 << 14, q8, sc8, t, kllm::PointwiseOp::kRelu);
	if (!st1.ok()) std::cout << "pipeline v2.1 int8 error: " << st1.message << "\n";
	std::cout << "Pipeline v2.1 int8: slabs=" << t.slabs_processed << ", ms_total=" << t.ms_total << "\n";
	kllm::PipelineTelemetry t2{};
	auto st2 = kllm::run_pipeline_v21_to_int4(x, 1 << 14, q4, sc4, t2, kllm::PointwiseOp::kRelu);
	if (!st2.ok()) std::cout << "pipeline v2.1 int4 error: " << st2.message << "\n";
	std::cout << "Pipeline v2.1 int4: slabs=" << t2.slabs_processed << ", ms_total=" << t2.ms_total << "\n";
}

int main() {
	test_fwht();
	test_ntt();
	test_countsketch();
	test_fused();
	test_fused_extra();
	test_simple_bias_relu();
	test_quant();
	test_rewards();
	test_ir();
	test_transforms_extras();
	test_config_and_onnx();
	test_training();
	test_nn_regression();
	test_pipeline_v21();
	return 0;
}