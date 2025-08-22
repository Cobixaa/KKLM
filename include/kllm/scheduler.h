#pragma once

#include <memory>
#include <vector>
#include <cstddef>

#include "kllm/ir.h"
#include "kllm/fast_transform.h"
#include "kllm/transform_extras.h"
#include "kllm/profiler.h"

namespace kllm {

struct ScheduledPlan {
	std::shared_ptr<Node> root;
	bool used_fusion = false;
};

class Scheduler {
public:
	explicit Scheduler(Profiler &p) : prof(p) {}

	ScheduledPlan plan(const std::shared_ptr<Node> &root) {
		ScheduledPlan sp;
		// For demo: if input tensor size is power-of-two and large, favor FWHT based fused path via existing Planner
		auto relu = std::dynamic_pointer_cast<ReluNode>(root);
		if (relu) {
			auto tr = std::dynamic_pointer_cast<TransformNode>(relu->input);
			if (tr) {
				sp.root = std::make_shared<FusedTransformReluNode>(tr->input);
				sp.used_fusion = true;
				return sp;
			}
		}
		sp.root = root;
		sp.used_fusion = false;
		return sp;
	}

private:
#if defined(__GNUC__) || defined(__clang__)
	[[maybe_unused]]
#endif
	Profiler &prof;
};

} // namespace kllm