#pragma once

#include <string>
#include "kllm/status.h"
#include "kllm/config.h"

namespace kllm {

inline Status import_onnx_model(const std::string &path) {
	(void)path;
	if (global_config().deterministic) {
		// In deterministic mode, skip random initializations, etc. Stub ok.
		return Status::OK();
	}
	return Status::OK();
}

} // namespace kllm