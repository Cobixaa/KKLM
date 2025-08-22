#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace kllm {

inline std::uint64_t splitmix64(std::uint64_t x) {
	x += 0x9e3779b97f4a7c15ull;
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
	x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
	x = x ^ (x >> 31);
	return x;
}

struct CountSketch {
	std::size_t sketch_size;
	std::size_t num_hashes;
	std::vector<std::uint64_t> seeds;

	explicit CountSketch(std::size_t sketch_size_, std::size_t num_hashes_, std::uint64_t seed_base = 0x12345678abcdef00ull)
		: sketch_size(sketch_size_) , num_hashes(num_hashes_), seeds(num_hashes_, 0) {
		for (std::size_t i = 0; i < num_hashes; ++i) {
			seeds[i] = splitmix64(seed_base + static_cast<std::uint64_t>(i) * 0x9e3779b97f4a7c15ull);
		}
	}

	inline void apply(const float *input, std::size_t length, float *output) const {
		if (input == nullptr || output == nullptr || sketch_size == 0 || num_hashes == 0) {
			return;
		}
		for (std::size_t i = 0; i < sketch_size; ++i) {
			output[i] = 0.0f;
		}
		for (std::size_t i = 0; i < length; ++i) {
			for (std::size_t h = 0; h < num_hashes; ++h) {
				const std::uint64_t mix = splitmix64(static_cast<std::uint64_t>(i) ^ seeds[h]);
				const std::size_t bucket = static_cast<std::size_t>(mix % static_cast<std::uint64_t>(sketch_size));
				const float sign = ((mix >> 63) == 0ull) ? 1.0f : -1.0f;
				output[bucket] += sign * input[i];
			}
		}
	}
};

} // namespace kllm