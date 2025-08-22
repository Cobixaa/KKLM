#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>

#if defined(__linux__)
	#include <pthread.h>
	#include <sched.h>
#endif

namespace kllm {

struct FreeDeleter {
	void operator()(void *ptr) const noexcept {
		std::free(ptr);
	}
};

inline std::unique_ptr<void, FreeDeleter> allocate_aligned_bytes(std::size_t size_bytes, std::size_t alignment) {
	if (alignment < alignof(void *)) {
		alignment = alignof(void *);
	}
	void *ptr = nullptr;
#if defined(_POSIX_VERSION)
	if (posix_memalign(&ptr, alignment, size_bytes) != 0) {
		return std::unique_ptr<void, FreeDeleter>(nullptr);
	}
	return std::unique_ptr<void, FreeDeleter>(ptr);
#else
	// Fallback to standard aligned_alloc if available (requires size to be multiple of alignment)
	std::size_t rounded = (size_bytes + alignment - 1u) / alignment * alignment;
	ptr = std::aligned_alloc(alignment, rounded);
	return std::unique_ptr<void, FreeDeleter>(ptr);
#endif
}

template <typename T>
inline std::unique_ptr<T, FreeDeleter> allocate_aligned(std::size_t count, std::size_t alignment) {
	const std::size_t total = count * sizeof(T);
	auto raw = allocate_aligned_bytes(total, alignment);
	return std::unique_ptr<T, FreeDeleter>(static_cast<T *>(raw.release()));
}

inline bool set_current_thread_affinity(int cpu_index) {
	(void)cpu_index;
#if defined(__linux__)
	if (cpu_index < 0) {
		return false;
	}
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(static_cast<unsigned int>(cpu_index), &set);
	const int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
	return result == 0;
#else
	return false;
#endif
}

} // namespace kllm