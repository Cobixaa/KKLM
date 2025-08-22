#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kllm/utils.h"
#include "kllm/quant.h"

namespace kllm {

// y = B x, where B is block-diagonal: blocks[i] is block i of size block_sizes[i] x block_sizes[i]
// Blocks are stored row-major, concatenated in blocks_data with offsets in offsets.
inline void block_diagonal_matvec(const float *blocks_data, const std::size_t *offsets, const std::size_t *block_sizes, std::size_t num_blocks,
	const float *x, float *y) {
	std::size_t x_offset = 0;
	std::size_t y_offset = 0;
	for (std::size_t b = 0; b < num_blocks; ++b) {
		const std::size_t n = block_sizes[b];
		const float *block = blocks_data + offsets[b];
		for (std::size_t i = 0; i < n; ++i) {
			float acc = 0.0f;
			const float *row = block + i * n;
			for (std::size_t j = 0; j < n; ++j) {
				acc += row[j] * x[x_offset + j];
			}
			y[y_offset + i] = acc;
		}
		x_offset += n;
		y_offset += n;
	}
}

// Mixed-precision path: int8 weights with per-block scale, int32 accumulate, output float
inline void block_diagonal_matvec_int8(const int8_t *blocks_q, const float *scales, const std::size_t *offsets, const std::size_t *block_sizes, std::size_t num_blocks,
	const float *x, float *y) {
	std::size_t x_offset = 0;
	std::size_t y_offset = 0;
	for (std::size_t b = 0; b < num_blocks; ++b) {
		const std::size_t n = block_sizes[b];
		const float s = scales[b];
		const int8_t *block = blocks_q + offsets[b];
		for (std::size_t i = 0; i < n; ++i) {
			int32_t acc_i32 = 0;
			const int8_t *row = block + i * n;
			for (std::size_t j = 0; j < n; ++j) {
				acc_i32 += static_cast<int32_t>(row[j]) * static_cast<int32_t>(static_cast<int>(x[x_offset + j]));
			}
			y[y_offset + i] = static_cast<float>(acc_i32) * s;
		}
		x_offset += n;
		y_offset += n;
	}
}

// Low-rank transform: y = U (V^T x), dims: U[out_dim x rank], V[in_dim x rank], V^T is rank x in_dim
inline void low_rank_apply(const float *U, const float *V, std::size_t out_dim, std::size_t in_dim, std::size_t rank,
	const float *x, float *y) {
	// t = V^T x => length rank
	std::vector<float> t(rank, 0.0f);
	for (std::size_t r = 0; r < rank; ++r) {
		float acc = 0.0f;
		for (std::size_t j = 0; j < in_dim; ++j) {
			acc += V[j * rank + r] * x[j];
		}
		t[r] = acc;
	}
	for (std::size_t i = 0; i < out_dim; ++i) {
		float acc = 0.0f;
		const float *u_row = U + i * rank;
		for (std::size_t r = 0; r < rank; ++r) {
			acc += u_row[r] * t[r];
		}
		y[i] = acc;
	}
}

} // namespace kllm