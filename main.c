#include "./mb.h"

// Algorithms under test

static void reverse_string(char *s) {
	if (s == NULL) return;
	const size_t len = strlen(s);
	if (len <= 1u) return;
	char *p = s;
	char *q = s + len - 1u;
	while (p < q) {
		const char t = *p;
		*p++ = *q;
		*q-- = t;
	}
}

static bool is_prime_u64(uint64_t n) {
	if (n <= 1u) return false;
	if (n <= 3u) return true;
	if ((n % 2u) == 0u || (n % 3u) == 0u) return false;
	for (uint64_t i = 5u; i <= n / i; i += 6u) {
		if ((n % i) == 0u || (n % (i + 2u)) == 0u) return false;
	}
	return true;
}

static int64_t fast_sum_i32(const int32_t * restrict arr, size_t count) {
	// Multiple accumulators reduce dependency chains and improve ILP
	int64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
	size_t i = 0u;
	for (; i + 4u <= count; i += 4u) {
		s0 += (int64_t)arr[i + 0u];
		s1 += (int64_t)arr[i + 1u];
		s2 += (int64_t)arr[i + 2u];
		s3 += (int64_t)arr[i + 3u];
	}
	int64_t acc = (s0 + s1) + (s2 + s3);
	for (; i < count; i++) {
		acc += (int64_t)arr[i];
	}
	return acc;
}

// --- Test helpers ---

static void test_reverse_string(void) {
	char buf[64];
	buf[0] = '\0';
	reverse_string(buf);
	MB_EXPECT_EQ_STR(buf, "");

	strcpy(buf, "a");
	reverse_string(buf);
	MB_EXPECT_EQ_STR(buf, "a");

	strcpy(buf, "ab");
	reverse_string(buf);
	MB_EXPECT_EQ_STR(buf, "ba");

	strcpy(buf, "abc");
	reverse_string(buf);
	MB_EXPECT_EQ_STR(buf, "cba");

	strcpy(buf, "abba");
	reverse_string(buf);
	MB_EXPECT_EQ_STR(buf, "abba");
}

static void test_is_prime_u64(void) {
	const uint64_t known_primes[] = {2u,3u,5u,7u,11u,13u,17u,19u,23u,29u,31u,37u,41u,43u,47u,53u,59u,61u,67u,71u,73u,79u,83u,89u,97u};
	for (size_t i = 0u; i < sizeof(known_primes)/sizeof(known_primes[0]); i++) {
		MB_EXPECT_TRUE(is_prime_u64(known_primes[i]));
	}
	for (uint64_t n = 0u; n <= 100u; n++) {
		bool is_p = is_prime_u64(n);
		bool truth = false;
		for (size_t i = 0u; i < sizeof(known_primes)/sizeof(known_primes[0]); i++) {
			if (known_primes[i] == n) { truth = true; break; }
		}
		MB_EXPECT_TRUE(is_p == truth);
	}
}

static void test_fast_sum_i32(void) {
	int32_t data1[] = {1, -2, 3, 4, -5, 6};
	MB_EXPECT_EQ_I64(fast_sum_i32(data1, sizeof(data1)/sizeof(data1[0])), 7);

	int32_t data2[100];
	for (size_t i = 0u; i < sizeof(data2)/sizeof(data2[0]); i++) {
		data2[i] = (int32_t)(i % 13) - 6; // values from -6..6
	}
	int64_t s = fast_sum_i32(data2, sizeof(data2)/sizeof(data2[0]));
	// compute expected
	int64_t expected = 0;
	for (size_t i = 0u; i < sizeof(data2)/sizeof(data2[0]); i++) {
		expected += (int64_t)data2[i];
	}
	MB_EXPECT_EQ_I64(s, expected);
}

// --- Benchmark wrappers ---

typedef struct ReverseCtx {
	char *dst;
	const char *src;
	size_t len;
} ReverseCtx;

static void reverse_wrapper(void *arg) {
	ReverseCtx *ctx = (ReverseCtx *)arg;
	memcpy(ctx->dst, ctx->src, ctx->len + 1u);
	reverse_string(ctx->dst);
}

typedef struct SumCtx {
	const int32_t *arr;
	size_t count;
	volatile int64_t sink; // prevent optimizing away
} SumCtx;

static void sum_wrapper(void *arg) {
	SumCtx *ctx = (SumCtx *)arg;
	ctx->sink = fast_sum_i32(ctx->arr, ctx->count);
}

typedef struct PrimeCtx {
	const uint64_t *nums;
	size_t count;
	size_t idx;
	volatile bool sink;
} PrimeCtx;

static void prime_wrapper(void *arg) {
	PrimeCtx *ctx = (PrimeCtx *)arg;
	uint64_t n = ctx->nums[ctx->idx];
	ctx->idx = (ctx->idx + 1u) % ctx->count;
	ctx->sink = is_prime_u64(n);
}

int main(void) {
	mb_test_reset();

	// Run tests
	test_reverse_string();
	test_is_prime_u64();
	test_fast_sum_i32();

	mb_test_summary();

	// Prepare data for benchmarks
	char *buf_big = (char *)malloc(1025u);
	char *buf_tmp = (char *)malloc(1025u);
	if (buf_big == NULL || buf_tmp == NULL) {
		fprintf(stderr, "OOM\n");
		free(buf_big);
		free(buf_tmp);
		return 1;
	}
	for (size_t i = 0u; i < 1024u; i++) {
		buf_big[i] = (char)('a' + (int)(i % 26u));
	}
	buf_big[1024u] = '\0';
	ReverseCtx rctx = { buf_tmp, buf_big, 1024u };

	const size_t arr_count = 4096u;
	int32_t *arr = (int32_t *)malloc(arr_count * sizeof(int32_t));
	if (arr == NULL) {
		fprintf(stderr, "OOM\n");
		free(buf_big);
		free(buf_tmp);
		return 1;
	}
	for (size_t i = 0u; i < arr_count; i++) {
		arr[i] = (int32_t)((i * 1103515245u + 12345u) & 0x7fffffffu) - 1073741824; // simple LCG-ish
	}
	SumCtx sctx = { arr, arr_count, 0 };

	const size_t prime_count = 4096u;
	uint64_t *nums = (uint64_t *)malloc(prime_count * sizeof(uint64_t));
	if (nums == NULL) {
		fprintf(stderr, "OOM\n");
		free(arr);
		free(buf_big);
		free(buf_tmp);
		return 1;
	}
	for (size_t i = 0u; i < prime_count; i++) {
		nums[i] = (uint64_t)(1000000u + (i * 97u));
	}
	PrimeCtx pctx = { nums, prime_count, 0u, false };

	// Choose iteration counts to get stable timing and report strictly in ns
	const size_t repeats = 15u;

	uint64_t med_ns, min_ns, max_ns;

	size_t it_rev = mb_calibrate_iters(reverse_wrapper, &rctx, 1u<<20, 2000000u);
	med_ns = mb_bench_ns(reverse_wrapper, &rctx, it_rev, repeats, &med_ns, &min_ns, &max_ns);
	printf("bench reverse_string(1024B): iters=%zu min=%" PRIu64 " ns, med=%" PRIu64 " ns, max=%" PRIu64 " ns per call\n", it_rev, min_ns, med_ns, max_ns);

	size_t it_sum = mb_calibrate_iters(sum_wrapper, &sctx, 1u<<20, 2000000u);
	med_ns = mb_bench_ns(sum_wrapper, &sctx, it_sum, repeats, &med_ns, &min_ns, &max_ns);
	printf("bench fast_sum_i32(4096): iters=%zu min=%" PRIu64 " ns, med=%" PRIu64 " ns, max=%" PRIu64 " ns per call\n", it_sum, min_ns, med_ns, max_ns);

	size_t it_prime = mb_calibrate_iters(prime_wrapper, &pctx, 1u<<20, 2000000u);
	med_ns = mb_bench_ns(prime_wrapper, &pctx, it_prime, repeats, &med_ns, &min_ns, &max_ns);
	printf("bench is_prime_u64(varied): iters=%zu min=%" PRIu64 " ns, med=%" PRIu64 " ns, max=%" PRIu64 " ns per call\n", it_prime, min_ns, med_ns, max_ns);

	free(nums);
	free(arr);
	free(buf_tmp);
	free(buf_big);

	return (g_mb_test.failed_asserts == 0u) ? 0 : 1;
}
