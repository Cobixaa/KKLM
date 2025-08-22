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
		kllm::ThreadPool pool(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4);
		auto t0 = clock_type::now();
		kllm::fwht_inplace_parallel(x.data(), x.size(), pool);
		auto t1 = clock_type::now();
		std::cout << "FWHT(par) 1M floats: " << time_ms(t0, t1) << " ms" << "\n";
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

	return 0;
}