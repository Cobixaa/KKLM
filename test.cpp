#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include "kklm.h"

static int failures = 0;

static void assert_near(float a, float b, float tol, const char *msg) {
	if (std::fabs(a - b) > tol || !std::isfinite(a) || !std::isfinite(b)) {
		std::cout << "ASSERT_NEAR failed: " << msg << " got=" << a << " expected=" << b << "\n";
		++failures;
	}
}

static void test_fwht_roundtrip() {
	std::vector<float> x(1 << 10);
	std::mt19937 rng(123);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (float &v : x) v = dist(rng);
	std::vector<float> y = x;
	kllm::fwht_inplace(y.data(), y.size());
	kllm::fwht_inplace_inverse(y.data(), y.size());
	for (std::size_t i = 0; i < x.size(); ++i) assert_near(y[i], x[i], 1e-4f, "fwht inverse");
}

static void test_ntt_roundtrip() {
	std::vector<std::uint32_t> a(1 << 10);
	for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<std::uint32_t>(i % 1000);
	std::vector<std::uint32_t> b = a;
	bool ok = kllm::ntt_inplace(b, false);
	bool ok2 = kllm::ntt_inplace(b, true);
	if (!(ok && ok2)) { std::cout << "NTT returned false\n"; ++failures; }
	for (std::size_t i = 0; i < a.size(); ++i) if (b[i] != a[i]) { ++failures; std::cout << "NTT mismatch at " << i << "\n"; break; }
}

static void test_countsketch_basic() {
	kllm::CountSketch cs(64, 3);
	std::vector<float> x(256, 1.0f);
	std::vector<float> y(64);
	cs.apply(x.data(), x.size(), y.data());
	float sum = 0.0f; for (float v : y) sum += v;
	if (!std::isfinite(sum)) { ++failures; std::cout << "CountSketch produced non-finite\n"; }
}

static void test_fused_paths() {
	std::vector<float> src(1024);
	std::vector<float> dst(1024, 0.0f);
	for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i) * 0.001f;
	kllm::fused_fwht_scale_add(src.data(), src.size(), 0.5f, dst.data());
	float checksum = 0.0f; for (float v : dst) checksum += v;
	if (!std::isfinite(checksum)) { ++failures; std::cout << "fused_fwht_scale_add non-finite\n"; }

	std::vector<float> bias(1024, 0.1f), out(1024);
	kllm::fused_fwht_bias_relu(src.data(), bias.data(), src.size(), out.data());
	for (float v : out) if (v < 0.0f) { ++failures; std::cout << "ReLU negative output\n"; break; }
}

static void test_quantization() {
	std::vector<float> x(1000);
	for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i) * 0.01f);
	kllm::QuantParams qp{}; std::vector<int8_t> q; std::vector<float> deq;
	auto st = kllm::quantize_dequantize(x, q, deq, qp);
	if (!st.ok()) { ++failures; std::cout << "quantize_dequantize status not OK\n"; }
	float max_abs_err = 0.0f; for (std::size_t i = 0; i < x.size(); ++i) max_abs_err = std::max(max_abs_err, std::fabs(x[i] - deq[i]));
	if (!(max_abs_err < 0.05f)) { ++failures; std::cout << "quantization error too high: " << max_abs_err << "\n"; }
}

static void test_ir_and_scheduler() {
	std::vector<float> x(1024, 1.0f);
	auto n0 = kllm::GraphBuilder::input(x);
	auto n1 = kllm::GraphBuilder::transform(n0);
	auto n2 = kllm::GraphBuilder::relu(n1);
	kllm::Profiler prof; kllm::Scheduler sched(prof);
	auto plan = sched.plan(n2);
	auto root = plan.root ? plan.root : n2;
	kllm::Tensor out = root->evaluate();
	float sum = 0.0f; for (float v : out.values) sum += v;
	if (!std::isfinite(sum)) { ++failures; std::cout << "IR output non-finite\n"; }
}

static void test_lowrank_and_blocks() {
	const std::size_t out_dim = 16, in_dim = 16, rank = 4;
	std::vector<float> U(out_dim * rank), V(in_dim * rank), x(in_dim), y(out_dim);
	std::mt19937 rng(7); std::uniform_real_distribution<float> d(-1.0f,1.0f);
	for (float &v : U) v = d(rng); for (float &v : V) v = d(rng); for (float &v : x) v = d(rng);
	kllm::low_rank_apply(U.data(), V.data(), out_dim, in_dim, rank, x.data(), y.data());
	float sum = 0.0f; for (float v : y) sum += v;
	if (!std::isfinite(sum)) { ++failures; std::cout << "low_rank non-finite\n"; }

	const std::size_t blocks = 4, bsz = 8; const std::size_t total = blocks * bsz;
	std::vector<std::size_t> offsets(blocks), sizes(blocks, bsz);
	std::vector<float> data(blocks * bsz * bsz);
	std::size_t off = 0; for (std::size_t b = 0; b < blocks; ++b) { offsets[b] = off; off += bsz * bsz; }
	for (float &v : data) v = d(rng);
	std::vector<float> xin(total), yout(total); for (float &v : xin) v = d(rng);
	kllm::block_diagonal_matvec(data.data(), offsets.data(), sizes.data(), blocks, xin.data(), yout.data());
	float sum2 = 0.0f; for (float v : yout) sum2 += v;
	if (!std::isfinite(sum2)) { ++failures; std::cout << "block_diag non-finite\n"; }
}

static void test_pipeline_v21_basic() {
	std::vector<float> x(1 << 16);
	std::mt19937 rng(17);
	std::uniform_real_distribution<float> d(-1.0f, 1.0f);
	for (float &v : x) v = d(rng);
	kllm::PipelineTelemetry t{}; std::vector<int8_t> q8; std::vector<float> sc;
	auto st = kllm::run_pipeline_v21_to_int8(x, 1 << 12, q8, sc, t, kllm::PointwiseOp::kRelu);
	if (!st.ok()) { ++failures; std::cout << "pipeline v2.1 status not OK\n"; }
	if (q8.empty() || sc.empty()) { ++failures; std::cout << "pipeline v2.1 empty outputs\n"; }
	float sum_sc = 0.0f; for (float v : sc) sum_sc += v; if (!std::isfinite(sum_sc)) { ++failures; std::cout << "pipeline v2.1 scales non-finite\n"; }
	kllm::set_deterministic(true);
	kllm::PipelineTelemetry t2{}; std::vector<int8_t> q8b; std::vector<float> scb;
	auto st2 = kllm::run_pipeline_v21_to_int8(x, 1 << 12, q8b, scb, t2, kllm::PointwiseOp::kRelu);
	if (!st2.ok()) { ++failures; std::cout << "pipeline v2.1 det status not OK\n"; }
	if (q8b.size() == 0 || scb.size() == 0) { ++failures; std::cout << "pipeline v2.1 det empty outputs\n"; }
	kllm::set_deterministic(false);
}

int main() {
	test_fwht_roundtrip();
	test_ntt_roundtrip();
	test_countsketch_basic();
	test_fused_paths();
	test_quantization();
	test_ir_and_scheduler();
	test_lowrank_and_blocks();
	test_pipeline_v21_basic();
	if (failures == 0) {
		std::cout << "ALL TESTS PASSED\n";
		return 0;
	}
	std::cout << failures << " failures\n";
	return 1;
}