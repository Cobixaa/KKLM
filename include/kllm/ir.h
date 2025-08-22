#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <functional>

#include "kllm/fast_transform.h"
#include "kllm/fused.h"

namespace kllm {

struct Tensor {
	std::vector<float> values;

	Tensor() = default;
	explicit Tensor(std::size_t n) : values(n, 0.0f) {}
	explicit Tensor(std::vector<float> v) : values(std::move(v)) {}

	std::size_t size() const { return values.size(); }
	float * data() { return values.data(); }
	const float * data() const { return values.data(); }
};

struct Node {
	virtual ~Node() = default;
	virtual Tensor evaluate() = 0;
};

struct InputNode : public Node {
	Tensor tensor;
	explicit InputNode(Tensor t) : tensor(std::move(t)) {}

	Tensor evaluate() override {
		return tensor;
	}
};

struct TransformNode : public Node {
	std::shared_ptr<Node> input;
	explicit TransformNode(std::shared_ptr<Node> in) : input(std::move(in)) {}

	Tensor evaluate() override {
		Tensor x = input->evaluate();
		fwht_inplace(x.data(), x.size());
		return x;
	}
};

struct ReluNode : public Node {
	std::shared_ptr<Node> input;
	explicit ReluNode(std::shared_ptr<Node> in) : input(std::move(in)) {}

	Tensor evaluate() override {
		Tensor x = input->evaluate();
		for (float &v : x.values) {
			if (v < 0.0f) v = 0.0f;
		}
		return x;
	}
};

// A very tiny planner that replaces Transform->Relu by a fused call at evaluation time.
struct FusedTransformReluNode : public Node {
	std::shared_ptr<Node> input;
	explicit FusedTransformReluNode(std::shared_ptr<Node> in) : input(std::move(in)) {}

	Tensor evaluate() override {
		Tensor x = input->evaluate();
		fwht_inplace(x.data(), x.size());
		for (float &v : x.values) {
			if (v < 0.0f) v = 0.0f;
		}
		return x;
	}
};

struct GraphBuilder {
	static std::shared_ptr<Node> input(const std::vector<float> &values) {
		return std::make_shared<InputNode>(Tensor(values));
	}
	static std::shared_ptr<Node> transform(const std::shared_ptr<Node> &in) {
		return std::make_shared<TransformNode>(in);
	}
	static std::shared_ptr<Node> relu(const std::shared_ptr<Node> &in) {
		return std::make_shared<ReluNode>(in);
	}
};

struct Planner {
	// If pattern is Transform(Relu(x)) or Relu(Transform(x)), use fused node.
	static std::shared_ptr<Node> plan(const std::shared_ptr<Node> &root) {
		// Only recognize ReluNode over TransformNode and vice versa for this demo.
		auto try_cast_relu = std::dynamic_pointer_cast<ReluNode>(root);
		if (try_cast_relu) {
			auto try_cast_transform = std::dynamic_pointer_cast<TransformNode>(try_cast_relu->input);
			if (try_cast_transform) {
				return std::make_shared<FusedTransformReluNode>(try_cast_transform->input);
			}
			return root;
		}
		auto try_cast_transform = std::dynamic_pointer_cast<TransformNode>(root);
		if (try_cast_transform) {
			auto try_cast_relu2 = std::dynamic_pointer_cast<ReluNode>(try_cast_transform->input);
			if (try_cast_relu2) {
				return std::make_shared<FusedTransformReluNode>(try_cast_relu2->input);
			}
			return root;
		}
		return root;
	}
};

} // namespace kllm