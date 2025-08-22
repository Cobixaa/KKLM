#ifndef MB_H_
#define MB_H_
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
// Minimal header-only micro test and benchmark utilities (nanosecond precision)
// Compatible with Linux and Termux aarch64. C11.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#if defined(__GNUC__) || defined(__clang__)
	#define MB_LIKELY(x)   (__builtin_expect(!!(x), 1))
	#define MB_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
	#define MB_LIKELY(x)   (x)
	#define MB_UNLIKELY(x) (x)
#endif

#ifndef MB_CLOCK_ID
	#ifdef CLOCK_MONOTONIC_RAW
		#define MB_CLOCK_ID CLOCK_MONOTONIC_RAW
	#else
		#define MB_CLOCK_ID CLOCK_MONOTONIC
	#endif
#endif

static inline uint64_t mb_now_ns(void) {
	struct timespec ts;
	if (MB_UNLIKELY(clock_gettime(MB_CLOCK_ID, &ts) != 0)) {
		// Fallback to CLOCK_MONOTONIC if RAW was not supported at runtime
		#ifdef CLOCK_MONOTONIC
			if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
				perror("clock_gettime");
				return 0u;
			}
		#else
			perror("clock_gettime");
			return 0u;
		#endif
	}
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void mb_sleep_ns(uint64_t ns) {
	struct timespec req;
	req.tv_sec = (time_t)(ns / 1000000000ull);
	req.tv_nsec = (long)(ns % 1000000000ull);
	while (nanosleep(&req, &req) != 0 && errno == EINTR) {
		// retry with remaining time
	}
}

// -------- Testing --------

typedef struct MbTestContext {
	size_t total_asserts;
	size_t failed_asserts;
} MbTestContext;

static MbTestContext g_mb_test = { 0u, 0u };

static inline void mb_test_reset(void) {
	g_mb_test.total_asserts = 0u;
	g_mb_test.failed_asserts = 0u;
}

static inline void mb_expect_true(bool condition, const char *expr, const char *file, int line) {
	g_mb_test.total_asserts++;
	if (MB_UNLIKELY(!condition)) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: %s\n", file, line, expr);
	}
}

static inline void mb_expect_eq_u64(uint64_t actual, uint64_t expected, const char *aexpr, const char *eexpr, const char *file, int line) {
	g_mb_test.total_asserts++;
	if (MB_UNLIKELY(actual != expected)) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: %s == %s (got=%" PRIu64 ", want=%" PRIu64 ")\n", file, line, aexpr, eexpr, actual, expected);
	}
}

static inline void mb_expect_eq_i64(int64_t actual, int64_t expected, const char *aexpr, const char *eexpr, const char *file, int line) {
	g_mb_test.total_asserts++;
	if (MB_UNLIKELY(actual != expected)) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: %s == %s (got=%" PRId64 ", want=%" PRId64 ")\n", file, line, aexpr, eexpr, actual, expected);
	}
}

static inline void mb_expect_eq_sz(size_t actual, size_t expected, const char *aexpr, const char *eexpr, const char *file, int line) {
	g_mb_test.total_asserts++;
	if (MB_UNLIKELY(actual != expected)) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: %s == %s (got=%zu, want=%zu)\n", file, line, aexpr, eexpr, actual, expected);
	}
}

static inline void mb_expect_eq_str(const char *actual, const char *expected, const char *aexpr, const char *eexpr, const char *file, int line) {
	g_mb_test.total_asserts++;
	const int cmp = (actual == NULL || expected == NULL) ? (actual == expected ? 0 : 1) : strcmp(actual, expected);
	if (MB_UNLIKELY(cmp != 0)) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: %s == %s (got=\"%s\", want=\"%s\")\n", file, line, aexpr, eexpr, actual ? actual : "(null)", expected ? expected : "(null)");
	}
}

#define MB_EXPECT_TRUE(expr) mb_expect_true((expr), #expr, __FILE__, __LINE__)
#define MB_EXPECT_EQ_U64(a, e) mb_expect_eq_u64((uint64_t)(a), (uint64_t)(e), #a, #e, __FILE__, __LINE__)
#define MB_EXPECT_EQ_I64(a, e) mb_expect_eq_i64((int64_t)(a), (int64_t)(e), #a, #e, __FILE__, __LINE__)
#define MB_EXPECT_EQ_SZ(a, e) mb_expect_eq_sz((size_t)(a), (size_t)(e), #a, #e, __FILE__, __LINE__)
#define MB_EXPECT_EQ_STR(a, e) mb_expect_eq_str((a), (e), #a, #e, __FILE__, __LINE__)

static inline void mb_test_summary(void) {
	const size_t total = g_mb_test.total_asserts;
	const size_t failed = g_mb_test.failed_asserts;
	const size_t passed = total - failed;
	printf("Tests: %zu total, %zu passed, %zu failed\n", total, passed, failed);
}

// -------- Benchmarking (nanoseconds) --------

static inline int mb_cmp_u64_asc(const void *lhs, const void *rhs) {
	const uint64_t a = *(const uint64_t *)lhs;
	const uint64_t b = *(const uint64_t *)rhs;
	if (a < b) return -1;
	if (a > b) return 1;
	return 0;
}

static inline uint64_t mb_bench_once_total_ns(void (*fn)(void *), void *arg, size_t iters) {
	// Measure total time for `iters` invocations of `fn(arg)`
	const uint64_t t0 = mb_now_ns();
	for (size_t i = 0; i < iters; i++) {
		fn(arg);
	}
	const uint64_t t1 = mb_now_ns();
	return t1 - t0;
}

static inline void mb_noop(void *arg) { (void)arg; }

// Returns median per-iteration ns; also fills optional min/max/median outputs
static inline uint64_t mb_bench_ns(void (*fn)(void *), void *arg, size_t iters, size_t repeats, uint64_t *out_median, uint64_t *out_min, uint64_t *out_max) {
	if (iters == 0u) { iters = 1u; }
	if (repeats == 0u) { repeats = 1u; }
	uint64_t *samples = (uint64_t *)malloc(repeats * sizeof(uint64_t));
	if (samples == NULL) { fprintf(stderr, "mb_bench_ns: OOM\n"); return 0u; }

	// Measure baseline overhead of the call loop using a no-op
	const uint64_t baseline_total = mb_bench_once_total_ns(mb_noop, NULL, iters);
	const uint64_t baseline_per_iter = baseline_total / iters;

	// Warmup once
	(void)mb_bench_once_total_ns(fn, arg, iters);

	for (size_t r = 0; r < repeats; r++) {
		const uint64_t total_ns = mb_bench_once_total_ns(fn, arg, iters);
		uint64_t per_iter = (total_ns / iters);
		if (per_iter > baseline_per_iter) {
			per_iter -= baseline_per_iter;
		} else {
			per_iter = 0u;
		}
		samples[r] = per_iter;
	}

	qsort(samples, repeats, sizeof(uint64_t), mb_cmp_u64_asc);
	const uint64_t min_v = samples[0];
	const uint64_t med_v = samples[repeats / 2u];
	const uint64_t max_v = samples[repeats - 1u];

	if (out_min) *out_min = min_v;
	if (out_median) *out_median = med_v;
	if (out_max) *out_max = max_v;

	free(samples);
	return med_v;
}

static inline size_t mb_calibrate_iters(void (*fn)(void *), void *arg, size_t max_iters, uint64_t target_total_ns) {
	if (max_iters == 0u) max_iters = 1u;
	if (target_total_ns == 0u) target_total_ns = 1000000u; // default 1ms total
	size_t iters = 1u;
	// Exponential search up to max_iters
	while (iters < max_iters) {
		uint64_t elapsed = mb_bench_once_total_ns(fn, arg, iters);
		if (elapsed >= target_total_ns) break;
		if (iters > (SIZE_MAX / 2u)) break;
		iters *= 2u;
	}
	if (iters > max_iters) iters = max_iters;
	if (iters == 0u) iters = 1u;
	return iters;
}

// ---- Extra asserts ----
static inline void mb_expect_ne_ptr(const void *a, const void *b, const char *aexpr, const char *bexpr, const char *file, int line) {
	g_mb_test.total_asserts++;
	if (MB_UNLIKELY(a == b)) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: %s != %s (both %p)\n", file, line, aexpr, bexpr, a);
	}
}
#define MB_EXPECT_NE_PTR(a,b) mb_expect_ne_ptr((a),(b),#a,#b,__FILE__,__LINE__)

static inline void mb_expect_eq_mem(const void *a, const void *b, size_t len, const char *aexpr, const char *bexpr, const char *file, int line) {
	g_mb_test.total_asserts++;
	if (MB_UNLIKELY(len > 0u && (a == NULL || b == NULL))) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: %s == %s (null with len=%zu)\n", file, line, aexpr, bexpr, len);
		return;
	}
	if (MB_UNLIKELY(memcmp(a, b, len) != 0)) {
		g_mb_test.failed_asserts++;
		fprintf(stderr, "[ASSERT FAIL] %s:%d: memory not equal for %s and %s (len=%zu)\n", file, line, aexpr, bexpr, len);
	}
}
#define MB_EXPECT_EQ_MEM(a,b,len) mb_expect_eq_mem((a),(b),(len),#a,#b,__FILE__,__LINE__)

// ---- Env helpers ----
static inline uint64_t mb_getenv_u64(const char *name, uint64_t default_value) {
	const char *s = getenv(name);
	if (s == NULL || *s == '\0') return default_value;
	char *endptr = NULL;
	unsigned long long v = strtoull(s, &endptr, 10);
	if (endptr == s) return default_value;
	return (uint64_t)v;
}

#endif // MB_H_
