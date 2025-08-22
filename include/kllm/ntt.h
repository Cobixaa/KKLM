#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>
#include <type_traits>

namespace kllm {

// 998244353 = 119 * 2^23 + 1, primitive root g = 3
static constexpr std::uint32_t kMod = 998244353u;
static constexpr std::uint32_t kPrimitiveRoot = 3u;

inline std::uint32_t add_mod(std::uint32_t a, std::uint32_t b) {
	std::uint32_t c = a + b;
	if (c >= kMod) c -= kMod;
	return c;
}

inline std::uint32_t sub_mod(std::uint32_t a, std::uint32_t b) {
	return (a >= b) ? (a - b) : (a + kMod - b);
}

inline std::uint32_t mul_mod(std::uint64_t a, std::uint64_t b) {
	return static_cast<std::uint32_t>((a * b) % kMod);
}

inline std::uint32_t pow_mod(std::uint32_t base, std::uint32_t exp) {
	std::uint64_t result = 1u;
	std::uint64_t cur = base;
	while (exp > 0) {
		if (exp & 1u) {
			result = (result * cur) % kMod;
		}
		cur = (cur * cur) % kMod;
		exp >>= 1u;
	}
	return static_cast<std::uint32_t>(result);
}

inline std::uint32_t inv_mod(std::uint32_t x) {
	// Fermat inverse: x^(mod-2)
	return pow_mod(x, kMod - 2u);
}

inline bool is_power_of_two(std::size_t value) {
	return value != 0 && (value & (value - 1)) == 0;
}

inline void bit_reverse_permute(std::vector<std::uint32_t> &a) {
	const std::size_t n = a.size();
	std::size_t j = 0;
	for (std::size_t i = 1; i < n; ++i) {
		std::size_t bit = n >> 1;
		for (; j & bit; bit >>= 1) {
			j ^= bit;
		}
		j ^= bit;
		if (i < j) {
			std::uint32_t tmp = a[i];
			a[i] = a[j];
			a[j] = tmp;
		}
	}
}

// In-place NTT over Z_kMod. Returns false if size is not power-of-two.
inline bool ntt_inplace(std::vector<std::uint32_t> &a, bool invert) {
	const std::size_t n = a.size();
	if (!is_power_of_two(n)) {
		return false;
	}
	bit_reverse_permute(a);

	for (std::size_t len = 2; len <= n; len <<= 1) {
		const std::size_t half = len >> 1;
		const std::uint32_t wlen = invert
			? inv_mod(pow_mod(kPrimitiveRoot, (kMod - 1u) / static_cast<std::uint32_t>(len)))
			: pow_mod(kPrimitiveRoot, (kMod - 1u) / static_cast<std::uint32_t>(len));
		for (std::size_t i = 0; i < n; i += len) {
			std::uint32_t w = 1u;
			for (std::size_t j = 0; j < half; ++j) {
				const std::uint32_t u = a[i + j];
				const std::uint32_t v = mul_mod(a[i + j + half], w);
				a[i + j] = add_mod(u, v);
				a[i + j + half] = sub_mod(u, v);
				w = mul_mod(w, wlen);
			}
		}
	}
	if (invert) {
		const std::uint32_t inv_n = inv_mod(static_cast<std::uint32_t>(n % kMod));
		for (std::size_t i = 0; i < n; ++i) {
			a[i] = mul_mod(a[i], inv_n);
		}
	}
	return true;
}

} // namespace kllm