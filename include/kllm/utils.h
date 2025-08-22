#pragma once

#include <cstddef>

namespace kllm {

inline bool is_power_of_two(std::size_t value) {
	return value != 0 && (value & (value - 1)) == 0;
}

#if defined(__GNUC__) || defined(__clang__)
	#define KLLM_LIKELY(x) (__builtin_expect(!!(x), 1))
	#define KLLM_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
	#define KLLM_LIKELY(x) (x)
	#define KLLM_UNLIKELY(x) (x)
#endif

#if defined(__GNUC__) || defined(__clang__)
	inline void prefetch(const void *ptr, int locality = 3) {
		if (locality <= 0) {
			__builtin_prefetch(ptr, 0, 0);
		} else if (locality == 1) {
			__builtin_prefetch(ptr, 0, 1);
		} else if (locality == 2) {
			__builtin_prefetch(ptr, 0, 2);
		} else {
			__builtin_prefetch(ptr, 0, 3);
		}
	}
#else
	inline void prefetch(const void *ptr, int locality = 3) {
		(void)ptr; (void)locality;
	}
#endif

#if defined(__GNUC__) || defined(__clang__)
	#define KLLM_ASSUME_ALIGNED(ptr, align) __builtin_assume_aligned((ptr), (align))
#else
	#define KLLM_ASSUME_ALIGNED(ptr, align) (ptr)
#endif

} // namespace kllm