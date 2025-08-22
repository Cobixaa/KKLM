#pragma once

#include <string>

namespace kllm {

enum class StatusCode {
	kOk = 0,
	kInvalidArgument = 1,
	kFailedPrecondition = 2,
	kInternal = 3
};

struct Status {
	StatusCode code;
	const char *message;

	constexpr bool ok() const { return code == StatusCode::kOk; }
	static constexpr Status OK() { return Status{StatusCode::kOk, "OK"}; }
};

inline Status make_status(StatusCode code, const char *msg) {
	return Status{code, msg};
}

} // namespace kllm