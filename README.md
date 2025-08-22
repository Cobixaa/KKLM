# Micro Bench (C11, single-header tests + nanosecond benchmarking)

## Build

- Linux/WSL/Termux:
```
make -C /workspace
```

- Optional CPU tuning:
```
EXTRA_CFLAGS="-mtune=native" make -C /workspace
```

## Run

- Default (text):
```
make -C /workspace run
```

- JSON (one object/line):
```
./build/mb_demo --format json --bench sum
```

- CSV (with fixed iterations and no tests):
```
./build/mb_demo --format csv --iters 10000 --no-tests
```

## Features
- Single header `mb.h` with:
  - Assertion helpers and summary
  - Nanosecond timing via `clock_gettime` (RAW/monotonic)
  - Calibration to reach target total runtime per benchmark
- Algorithms + tests:
  - `reverse_string`, `is_prime_u64`, `fast_sum_i32` (optimized)
  - `to_lowercase_inplace`, `memxor_u8`, `fnv1a_64`
- Outputs: text, JSON, CSV
- Strict build: C11, `-Wall -Wextra -Werror -pedantic` (0 warnings)
- Termux aarch64 friendly

## CLI
```
./build/mb_demo [--format text|json|csv] [--repeats N] [--target-ns N] [--iters N]
                [--bench NAME] [--no-tests] [--list]
```
- `--bench` NAME in {reverse,sum,prime,lower,memxor,fnv1a}
- `--target-ns` is the total runtime per repeat (e.g., 2000000 for ~2ms)
- `--iters` overrides calibration

## How nanosecond timing is achieved
- Uses POSIX `clock_gettime` with `CLOCK_MONOTONIC_RAW` if available (fallback to `CLOCK_MONOTONIC`).
- Converts timespec to uint64 nanoseconds: `ns = sec*1e9 + nsec`.
- Measures loop total time across many iterations, subtracts baseline loop overhead measured via a no-op, then reports per-call ns.
- Median of repeated samples is used to reduce noise; min/max reported as well.

## Termux notes (aarch64)
- Termux provides `clang` with POSIX support; build as usual:
```
make
```
- Optionally pass CPU tuning flags:
```
EXTRA_CFLAGS="-march=armv8-a -mtune=native" make
```

## Will other headers be deleted?
- No. This project is intentionally single-header for the framework (`mb.h`). There are no extra headers to delete. If you add more headers later, keep them; `mb.h` remains the only required one for tests/bench.