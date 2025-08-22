#pragma once

#include <cstddef>

namespace kllm {

struct Config {
	bool deterministic = false;
	std::size_t num_threads = 0; // 0 => use hardware_concurrency
};

inline Config & global_config() {
	static Config cfg{};
	return cfg;
}

inline void set_deterministic(bool enabled) {
	global_config().deterministic = enabled;
}

inline void set_num_threads(std::size_t n) {
	global_config().num_threads = n;
}

} // namespace kllm