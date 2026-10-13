# microtensor

> A tiny, **heap-free** C++ tensor and **int8 quantized-inference** core — the
> math that lets neural nets run on constrained, edge / microcontroller-class
> hardware, written from first principles with zero dependencies.

[![CI](https://github.com/michaelnwanna/microtensor/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This is part of a low-level C++ portfolio focused on performance and
determinism. It complements
[`mcu-baremetal`](https://github.com/michaelnwanna/mcu-baremetal): the same
no-heap, no-hidden-cost discipline, applied to inference math.

## What's inside

- **`Matrix<T, Rows, Cols>`** — dimensions are template parameters, so storage is
  a single `std::array` with **no dynamic allocation**. Cache-friendly `matmul`
  and `transpose`, all `constexpr`-friendly.
- **Symmetric int8 quantization** — `quantize` / `dequantize` with a single float
  scale and round-half-away-from-zero saturation.
- **Integer matmul kernel** — int8 operands, **int32 accumulation**, one final
  dequantize. This is exactly how TFLite-Micro / CMSIS-NN kernels work, distilled
  to its essence.

## Quick look

```cpp
#include "microtensor/matrix.hpp"
#include "microtensor/quantize.hpp"

mt::Matrix<float, 2, 3> a = /* ... */;
mt::Matrix<float, 3, 2> b = /* ... */;

mt::Matrix<int8_t, 2, 3> qa;
mt::Matrix<int8_t, 3, 2> qb;
auto pa = mt::quantizeMatrix(a, qa);          // pick scales, quantize
auto pb = mt::quantizeMatrix(b, qb);

auto out = mt::quantizedMatmul(qa, pa, qb, pb); // int8 in, float out
```

See [`examples/quantized_matmul.cpp`](examples/quantized_matmul.cpp), which
prints the int8 result alongside the exact float reference and the maximum error.

## Build & test

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
./build/quantized_matmul
```

Requires CMake ≥ 3.20 and a C++17 compiler. Tests are dependency-free.

## Roadmap

- Per-axis (per-channel) quantization and asymmetric zero-points.
- ReLU / requantization fused into the kernel.
- A minimal sequential model runner (dense → activation → dense).
- SIMD (NEON / AVX) and a CMSIS-NN-style path for Cortex-M, so models from this
  core can drop straight onto the `mcu-baremetal` targets.

## License

MIT — see [`LICENSE`](LICENSE).
