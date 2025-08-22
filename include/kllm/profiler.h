#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

namespace kllm {

struct KernelTiming {
	double ms;
	std::size_t bytes;
};

class Profiler {
public:
	void record(const std::string &name, double ms, std::size_t bytes = 0) {
		db[name] = KernelTiming{ms, bytes};
	}

	KernelTiming get(const std::string &name) const {
		auto it = db.find(name);
		if (it == db.end()) return KernelTiming{0.0, 0};
		return it->second;
	}

private:
	std::unordered_map<std::string, KernelTiming> db;
};

struct ScopeTimer {
	std::chrono::steady_clock::time_point start;
	Profiler *prof;
	std::string name;
	std::size_t bytes;

	ScopeTimer(Profiler &p, std::string n, std::size_t b = 0)
		: start(std::chrono::steady_clock::now()), prof(&p), name(std::move(n)), bytes(b) {}
	~ScopeTimer() {
		auto end = std::chrono::steady_clock::now();
		double ms = std::chrono::duration<double, std::milli>(end - start).count();
		if (prof) prof->record(name, ms, bytes);
	}
};

} // namespace kllm