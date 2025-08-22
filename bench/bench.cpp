#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

#include "kllm/kllm.h"

using clock_type = std::chrono::steady_clock;

static double time_ms(clock_type::time_point a, clock_type::time_point b) {
	return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	// FWHT bench
	{
		const std::size_t n = 1 << 20; // 1M
		std::vector<float> x(n);
		for (float &v : x) v = dist(rng);
		auto t0 = clock_type::now();
		kllm::fwht_inplace(x.data(), x.size());
		auto t1 = clock_type::now();
		std::cout << "FWHT 1M floats: " << time_ms(t0, t1) << " ms" << "\n";
	}

	// Parallel FWHT bench
	{
		const std::size_t n = 1 << 20;
		std::vector<float> x(n);
		for (float &v : x) v = dist(rng);
		std::size_t threads = kllm::global_config().num_threads ? kllm::global_config().num_threads : (std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		kllm::ThreadPool pool(threads);
		auto t0 = clock_type::now();
		kllm::fwht_inplace_parallel(x.data(), x.size(), pool);
		auto t1 = clock_type::now();
		std::cout << "FWHT(par," << threads << ") 1M floats: " << time_ms(t0, t1) << " ms" << "\n";
	}

	// Fused FWHT-scale-add
	{
		const std::size_t n = 1 << 20;
		std::vector<float> x(n), y(n, 0.0f);
		for (float &v : x) v = dist(rng);
		auto t0 = clock_type::now();
		kllm::fused_fwht_scale_add(x.data(), x.size(), 0.25f, y.data());
		auto t1 = clock_type::now();
		std::cout << "Fused FWHT-scale-add 1M: " << time_ms(t0, t1) << " ms" << "\n";
	}

	// NTT bench
	{
		const std::size_t n = 1 << 18; // 262,144
		std::vector<std::uint32_t> a(n);
		for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<std::uint32_t>(i % 1000);
		auto t0 = clock_type::now();
		kllm::ntt_inplace(a, false);
		auto t1 = clock_type::now();
		std::cout << "NTT 262k uint32: " << time_ms(t0, t1) << " ms" << "\n";
	}

	// CountSketch bench
	{
		const std::size_t n = 1 << 20;
		std::vector<float> x(n);
		for (float &v : x) v = dist(rng);
		kllm::CountSketch cs(1 << 18, 3);
		std::vector<float> out(1 << 18);
		auto t0 = clock_type::now();
		cs.apply(x.data(), x.size(), out.data());
		auto t1 = clock_type::now();
		std::cout << "CountSketch 1M -> 262k (3 hashes): " << time_ms(t0, t1) << " ms" << "\n";
	}

	// Block-diagonal float bench: 1024 blocks of 16x16 => 16384 total
	{
		const std::size_t blocks = 1024;
		const std::size_t bsz = 16;
		const std::size_t total = blocks * bsz;
		std::vector<std::size_t> offsets(blocks);
		std::vector<std::size_t> sizes(blocks, bsz);
		std::vector<float> data(blocks * bsz * bsz);
		std::size_t off = 0;
		for (std::size_t b = 0; b < blocks; ++b) { offsets[b] = off; off += bsz * bsz; }
		for (float &v : data) v = dist(rng);
		std::vector<float> x(total), y(total);
		for (float &v : x) v = dist(rng);
		auto t0 = clock_type::now();
		kllm::block_diagonal_matvec(data.data(), offsets.data(), sizes.data(), blocks, x.data(), y.data());
		auto t1 = clock_type::now();
		std::cout << "BlockDiag float 1024x(16x16): " << time_ms(t0, t1) << " ms" << "\n";
	}

	// Block-diagonal int8 bench (mixed precision)
	{
		const std::size_t blocks = 1024;
		const std::size_t bsz = 16;
		const std::size_t total = blocks * bsz;
		std::vector<std::size_t> offsets(blocks);
		std::vector<std::size_t> sizes(blocks, bsz);
		std::vector<float> data(blocks * bsz * bsz);
		std::size_t off = 0;
		for (std::size_t b = 0; b < blocks; ++b) { offsets[b] = off; off += bsz * bsz; }
		for (float &v : data) v = dist(rng);
		std::vector<int8_t> q(off);
		std::vector<float> scales(blocks, 0.1f);
		// naive quant per block
		std::size_t base = 0;
		for (std::size_t b = 0; b < blocks; ++b) {
			float max_abs = 0.0f;
			for (std::size_t i = 0; i < bsz*bsz; ++i) max_abs = std::max(max_abs, std::fabs(data[base + i]));
			float s = (max_abs <= 1e-8f) ? 1.0f : (max_abs / 127.0f);
			if (!(s > 0.0f)) s = 1.0f;
			scales[b] = s;
			for (std::size_t i = 0; i < bsz*bsz; ++i) {
				int v = static_cast<int>(data[base + i] / s + (data[base + i] >= 0.0f ? 0.5f : -0.5f));
				if (v > 127) v = 127; if (v < -127) v = -127;
				q[base + i] = static_cast<int8_t>(v);
			}
			base += bsz * bsz;
		}
		std::vector<float> x(total), y(total);
		for (float &v : x) v = dist(rng);
		auto t0 = clock_type::now();
		kllm::block_diagonal_matvec_int8(q.data(), scales.data(), offsets.data(), sizes.data(), blocks, x.data(), y.data());
		auto t1 = clock_type::now();
		std::cout << "BlockDiag int8 1024x(16x16): " << time_ms(t0, t1) << " ms" << "\n";
	}

	// Low-rank bench: out=4096, in=4096, rank=64
	{
		const std::size_t out_dim = 4096, in_dim = 4096, rank = 64;
		std::vector<float> U(out_dim * rank), V(in_dim * rank), x(in_dim), y(out_dim);
		for (float &v : U) v = dist(rng);
		for (float &v : V) v = dist(rng);
		for (float &v : x) v = dist(rng);
		auto t0 = clock_type::now();
		kllm::low_rank_apply(U.data(), V.data(), out_dim, in_dim, rank, x.data(), y.data());
		auto t1 = clock_type::now();
		std::cout << "LowRank 4096x4096 (r=64): " << time_ms(t0, t1) << " ms" << "\n";
	}

	return 0;
}