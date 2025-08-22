#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <cstddef>

namespace kllm {

class ThreadPool {
public:
	explicit ThreadPool(std::size_t num_threads) : stop_flag(false), outstanding_tasks(0) {
		if (num_threads == 0) num_threads = 1;
		workers.reserve(num_threads);
		for (std::size_t i = 0; i < num_threads; ++i) {
			workers.emplace_back([this]() { this->worker_loop(); });
		}
	}

	~ThreadPool() {
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			stop_flag = true;
		}
		cv.notify_all();
		for (auto &t : workers) {
			if (t.joinable()) t.join();
		}
	}

	void enqueue(std::function<void()> fn) {
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			++outstanding_tasks;
			tasks.push(std::move(fn));
		}
		cv.notify_one();
	}

	void wait() {
		std::unique_lock<std::mutex> lock(done_mutex);
		done_cv.wait(lock, [this]() { return outstanding_tasks.load() == 0; });
	}

	std::size_t size() const { return workers.size(); }

private:
	void worker_loop() {
		for (;;) {
			std::function<void()> task;
			{
				std::unique_lock<std::mutex> lock(queue_mutex);
				cv.wait(lock, [this]() { return stop_flag || !tasks.empty(); });
				if (stop_flag && tasks.empty()) return;
				task = std::move(tasks.front());
				tasks.pop();
			}
			task();
			if (outstanding_tasks.fetch_sub(1) == 1) {
				std::lock_guard<std::mutex> g(done_mutex);
				done_cv.notify_all();
			}
		}
	}

	std::vector<std::thread> workers;
	std::queue<std::function<void()>> tasks;
	mutable std::mutex queue_mutex;
	std::condition_variable cv;
	std::atomic<bool> stop_flag;
	std::atomic<std::size_t> outstanding_tasks;
	std::condition_variable done_cv;
	std::mutex done_mutex;
};

inline void parallel_for_blocks(ThreadPool &pool, std::size_t begin, std::size_t end, std::size_t step, std::function<void(std::size_t)> fn) {
	if (begin >= end || step == 0) return;
	// Chunk by number of workers
	const std::size_t num_workers = pool.size();
	const std::size_t num_iters = (end - begin + step - 1) / step;
	const std::size_t chunk_iters = (num_iters + num_workers - 1) / num_workers;
	std::size_t start_iter = 0;
	while (start_iter < num_iters) {
		const std::size_t this_iters = (chunk_iters < (num_iters - start_iter)) ? chunk_iters : (num_iters - start_iter);
		const std::size_t base = begin + start_iter * step;
		pool.enqueue([=]() {
			for (std::size_t j = 0; j < this_iters; ++j) {
				fn(base + j * step);
			}
		});
		start_iter += this_iters;
	}
	pool.wait();
}

} // namespace kllm