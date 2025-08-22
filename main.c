#include "./mb.h"

// ---- CLI config and output helpers ----

typedef enum OutputFormat {
	OUTPUT_TEXT = 0,
	OUTPUT_JSON = 1, // NDJSON (one JSON object per line)
	OUTPUT_CSV  = 2
} OutputFormat;

typedef struct AppConfig {
	OutputFormat format;
	size_t repeats;
	uint64_t target_total_ns;
	size_t iters_override;
	bool run_tests;
	bool bench_reverse;
	bool bench_sum;
	bool bench_prime;
	bool bench_lower;
	bool bench_memxor;
	bool bench_fnv1a;
} AppConfig;

static void app_config_init_defaults(AppConfig *cfg) {
	cfg->format = OUTPUT_TEXT;
	cfg->repeats = 15u;
	cfg->target_total_ns = 2000000u; // 2ms
	cfg->iters_override = 0u;
	cfg->run_tests = true;
	cfg->bench_reverse = true;
	cfg->bench_sum = true;
	cfg->bench_prime = true;
	cfg->bench_lower = true;
	cfg->bench_memxor = true;
	cfg->bench_fnv1a = true;
}

static bool parse_size_t_arg(const char *s, size_t *out_value) {
	if (s == NULL || *s == '\0') return false;
	char *endptr = NULL;
	unsigned long long v = strtoull(s, &endptr, 10);
	if (endptr == s || *endptr != '\0') return false;
	*out_value = (size_t)v;
	return true;
}

static bool parse_u64_arg(const char *s, uint64_t *out_value) {
	if (s == NULL || *s == '\0') return false;
	char *endptr = NULL;
	unsigned long long v = strtoull(s, &endptr, 10);
	if (endptr == s || *endptr != '\0') return false;
	*out_value = (uint64_t)v;
	return true;
}

static bool parse_format(const char *s, OutputFormat *out_fmt) {
	if (s == NULL) return false;
	if (strcmp(s, "text") == 0) { *out_fmt = OUTPUT_TEXT; return true; }
	if (strcmp(s, "json") == 0) { *out_fmt = OUTPUT_JSON; return true; }
	if (strcmp(s, "csv") == 0)  { *out_fmt = OUTPUT_CSV;  return true; }
	return false;
}

static void print_usage(const char *prog) {
	printf("Usage: %s [--format text|json|csv] [--repeats N] [--target-ns N] [--iters N]\\n", prog);
	printf("       [--bench NAME]... [--no-tests] [--list]\\n");
	printf("Benches: reverse, sum, prime, lower, memxor, fnv1a\\n");
}

static void parse_args(int argc, char **argv, AppConfig *cfg) {
	for (int i = 1; i < argc; i++) {
		const char *arg = argv[i];
		if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
			print_usage(argv[0]);
			exit(0);
		} else if (strcmp(arg, "--format") == 0) {
			if (i + 1 < argc) {
				OutputFormat f;
				if (parse_format(argv[i+1], &f)) { cfg->format = f; }
				i++;
			}
		} else if (strcmp(arg, "--repeats") == 0) {
			if (i + 1 < argc) { (void)parse_size_t_arg(argv[i+1], &cfg->repeats); i++; }
		} else if (strcmp(arg, "--target-ns") == 0) {
			if (i + 1 < argc) { (void)parse_u64_arg(argv[i+1], &cfg->target_total_ns); i++; }
		} else if (strcmp(arg, "--iters") == 0) {
			if (i + 1 < argc) { (void)parse_size_t_arg(argv[i+1], &cfg->iters_override); i++; }
		} else if (strcmp(arg, "--no-tests") == 0) {
			cfg->run_tests = false;
		} else if (strcmp(arg, "--bench") == 0) {
			if (i + 1 < argc) {
				const char *name = argv[i+1];
				cfg->bench_reverse = cfg->bench_sum = cfg->bench_prime = cfg->bench_lower = cfg->bench_memxor = cfg->bench_fnv1a = false;
				if (strcmp(name, "reverse") == 0) cfg->bench_reverse = true;
				else if (strcmp(name, "sum") == 0) cfg->bench_sum = true;
				else if (strcmp(name, "prime") == 0) cfg->bench_prime = true;
				else if (strcmp(name, "lower") == 0) cfg->bench_lower = true;
				else if (strcmp(name, "memxor") == 0) cfg->bench_memxor = true;
				else if (strcmp(name, "fnv1a") == 0) cfg->bench_fnv1a = true;
				i++;
			}
		} else if (strcmp(arg, "--list") == 0) {
			printf("reverse\nsum\nprime\nlower\nmemxor\nfnv1a\n");
			exit(0);
		}
	}
}

static void print_bench_result(const AppConfig *cfg, const char *name, size_t iters, uint64_t min_ns, uint64_t med_ns, uint64_t max_ns) {
	static bool csv_header_printed = false;
	switch (cfg->format) {
		case OUTPUT_TEXT:
			printf("bench %s: iters=%zu min=%" PRIu64 " ns, med=%" PRIu64 " ns, max=%" PRIu64 " ns per call\n", name, iters, min_ns, med_ns, max_ns);
			break;
		case OUTPUT_JSON:
			printf("{\"name\":\"%s\",\"iters\":%zu,\"min_ns\":%" PRIu64 ",\"med_ns\":%" PRIu64 ",\"max_ns\":%" PRIu64 "}\n", name, iters, min_ns, med_ns, max_ns);
			break;
		case OUTPUT_CSV:
			if (!csv_header_printed) {
				printf("name,iters,min_ns,med_ns,max_ns\n");
				csv_header_printed = true;
			}
			printf("%s,%zu,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", name, iters, min_ns, med_ns, max_ns);
			break;
	}
}

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

static void to_lowercase_inplace(char *s) {
	if (s == NULL) return;
	for (char *p = s; *p != '\0'; ++p) {
		unsigned char c = (unsigned char)*p;
		if (c >= 'A' && c <= 'Z') {
			*p = (char)(c + ('a' - 'A'));
		}
	}
}

static void memxor_u8(uint8_t * restrict dst, const uint8_t * restrict a, const uint8_t * restrict b, size_t len) {
	for (size_t i = 0u; i < len; i++) {
		dst[i] = (uint8_t)(a[i] ^ b[i]);
	}
}

#define FNV1A64_OFFSET 1469598103934665603ull
#define FNV1A64_PRIME  1099511628211ull
static uint64_t fnv1a_64(const uint8_t *data, size_t len) {
	uint64_t hash = (uint64_t)FNV1A64_OFFSET;
	for (size_t i = 0u; i < len; i++) {
		hash ^= (uint64_t)data[i];
		hash *= (uint64_t)FNV1A64_PRIME;
	}
	return hash;
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

// --- Extended tests ---
static void test_to_lowercase_inplace(void) {
	char buf[64];
	strcpy(buf, "AbC xyz 123 !");
	to_lowercase_inplace(buf);
	MB_EXPECT_EQ_STR(buf, "abc xyz 123 !");
}

static void test_memxor_u8(void) {
	uint8_t a[8] = {0x00,0xFF,0x55,0xAA,0x0F,0xF0,0x12,0x34};
	uint8_t b[8] = {0xFF,0x00,0xAA,0x55,0xF0,0x0F,0x34,0x12};
	uint8_t out[8];
	memxor_u8(out, a, b, 8u);
	uint8_t exp[8] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0x26,0x26};
	MB_EXPECT_EQ_MEM(out, exp, 8u);
}

static void test_fnv1a_64(void) {
	uint64_t h_empty = fnv1a_64((const uint8_t *)"", 0u);
	MB_EXPECT_EQ_U64(h_empty, (uint64_t)FNV1A64_OFFSET);
	uint64_t h1 = fnv1a_64((const uint8_t *)"hello", 5u);
	uint64_t h2 = fnv1a_64((const uint8_t *)"world", 5u);
	MB_EXPECT_TRUE(h1 != h2);
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

// --- New benchmark wrappers ---
typedef struct LowerCtx { char *dst; const char *src; size_t len; } LowerCtx;
static void lower_wrapper(void *arg) {
	LowerCtx *ctx = (LowerCtx *)arg;
	memcpy(ctx->dst, ctx->src, ctx->len + 1u);
	to_lowercase_inplace(ctx->dst);
}

typedef struct MemxorCtx { uint8_t *dst; const uint8_t *a; const uint8_t *b; size_t len; volatile uint8_t sink; } MemxorCtx;
static void memxor_wrapper(void *arg) {
	MemxorCtx *ctx = (MemxorCtx *)arg;
	memxor_u8(ctx->dst, ctx->a, ctx->b, ctx->len);
	ctx->sink = ctx->dst[0];
}

typedef struct HashCtx { const uint8_t *data; size_t len; volatile uint64_t sink; } HashCtx;
static void fnv1a_wrapper(void *arg) {
	HashCtx *ctx = (HashCtx *)arg;
	ctx->sink = fnv1a_64(ctx->data, ctx->len);
}

int main(int argc, char **argv) {
	mb_test_reset();

	AppConfig cfg; app_config_init_defaults(&cfg);
	parse_args(argc, argv, &cfg);

	// Run tests (optional)
	if (cfg.run_tests) {
		test_reverse_string();
		test_is_prime_u64();
		test_fast_sum_i32();
		test_to_lowercase_inplace();
		test_memxor_u8();
		test_fnv1a_64();
		if (cfg.format == OUTPUT_TEXT) {
			mb_test_summary();
		}
	}

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
	const size_t repeats = cfg.repeats;

	uint64_t med_ns, min_ns, max_ns;

	size_t it_rev = (cfg.iters_override ? cfg.iters_override : mb_calibrate_iters(reverse_wrapper, &rctx, 1u<<20, cfg.target_total_ns));
	med_ns = mb_bench_ns(reverse_wrapper, &rctx, it_rev, repeats, &med_ns, &min_ns, &max_ns);
	if (cfg.bench_reverse) print_bench_result(&cfg, "reverse_string(1024B)", it_rev, min_ns, med_ns, max_ns);

	size_t it_sum = (cfg.iters_override ? cfg.iters_override : mb_calibrate_iters(sum_wrapper, &sctx, 1u<<20, cfg.target_total_ns));
	med_ns = mb_bench_ns(sum_wrapper, &sctx, it_sum, repeats, &med_ns, &min_ns, &max_ns);
	if (cfg.bench_sum) print_bench_result(&cfg, "fast_sum_i32(4096)", it_sum, min_ns, med_ns, max_ns);

	size_t it_prime = (cfg.iters_override ? cfg.iters_override : mb_calibrate_iters(prime_wrapper, &pctx, 1u<<20, cfg.target_total_ns));
	med_ns = mb_bench_ns(prime_wrapper, &pctx, it_prime, repeats, &med_ns, &min_ns, &max_ns);
	if (cfg.bench_prime) print_bench_result(&cfg, "is_prime_u64(varied)", it_prime, min_ns, med_ns, max_ns);

	// New benches
	LowerCtx lw = { buf_tmp, buf_big, 1024u };
	size_t it_lower = (cfg.iters_override ? cfg.iters_override : mb_calibrate_iters(lower_wrapper, &lw, 1u<<20, cfg.target_total_ns));
	med_ns = mb_bench_ns(lower_wrapper, &lw, it_lower, repeats, &med_ns, &min_ns, &max_ns);
	if (cfg.bench_lower) print_bench_result(&cfg, "to_lowercase_inplace(1024B)", it_lower, min_ns, med_ns, max_ns);

	const size_t xor_len = 4096u;
	uint8_t *xa = (uint8_t *)malloc(xor_len);
	uint8_t *xb = (uint8_t *)malloc(xor_len);
	uint8_t *xd = (uint8_t *)malloc(xor_len);
	if (xa == NULL || xb == NULL || xd == NULL) {
		fprintf(stderr, "OOM\n");
		free(xa); free(xb); free(xd);
		free(nums); free(arr); free(buf_tmp); free(buf_big);
		return 1;
	}
	for (size_t i = 0u; i < xor_len; i++) { xa[i] = (uint8_t)(i & 0xFFu); xb[i] = (uint8_t)((i * 37u) & 0xFFu); }
	MemxorCtx mx = { xd, xa, xb, xor_len, 0 };
	size_t it_memxor = (cfg.iters_override ? cfg.iters_override : mb_calibrate_iters(memxor_wrapper, &mx, 1u<<20, cfg.target_total_ns));
	med_ns = mb_bench_ns(memxor_wrapper, &mx, it_memxor, repeats, &med_ns, &min_ns, &max_ns);
	if (cfg.bench_memxor) print_bench_result(&cfg, "memxor_u8(4096B)", it_memxor, min_ns, med_ns, max_ns);

	HashCtx hx = { (const uint8_t *)buf_big, 1024u, 0u };
	size_t it_hash = (cfg.iters_override ? cfg.iters_override : mb_calibrate_iters(fnv1a_wrapper, &hx, 1u<<20, cfg.target_total_ns));
	med_ns = mb_bench_ns(fnv1a_wrapper, &hx, it_hash, repeats, &med_ns, &min_ns, &max_ns);
	if (cfg.bench_fnv1a) print_bench_result(&cfg, "fnv1a_64(1024B)", it_hash, min_ns, med_ns, max_ns);

	free(xd); free(xb); free(xa);

	free(nums);
	free(arr);
	free(buf_tmp);
	free(buf_big);

	return (g_mb_test.failed_asserts == 0u) ? 0 : 1;
}
