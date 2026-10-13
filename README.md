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
- **Heap-free inference runtime** — statically sized dense layers, int8
  requantization, ReLU, ArgMax, and a two-layer sequential classifier whose
  dimensions are checked at compile time.

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

## From a kernel to a model

[`examples/sensor_classifier.cpp`](examples/sensor_classifier.cpp) connects the
individual operations into a complete vibration classifier:

```text
four int8 sensor features
  |
  v
Dense<4, 5> -> ReLU -> Dense<5, 3> -> ArgMax
           |
           v
           normal / imbalance / impact
```

Every size is part of the C++ type. Connecting a five-element hidden tensor to a
layer that expects six elements is therefore a compile error, not a runtime
surprise. Weights and biases live inside the model object, while hidden values
and scores live inside the returned result. Neither path uses `new`, `malloc`,
exceptions, RTTI, or a hidden allocator.

### The integer arithmetic

A dense neuron starts with an `int32` bias and accumulates products of `int8`
values:

```text
accumulator = bias + sum(input[i] * weight[i])
```

Why is the accumulator wider? One int8 product can reach `127 * 127 = 16129`;
adding many of those in another int8 would overflow almost immediately. After
the dot product, `requantize` converts the accumulator to the next layer's int8
scale:

```text
real value = accumulator * inputScale * weightScale
output int8 = round(real value / outputScale)

multiplier = inputScale * weightScale / outputScale
output int8 = saturate(round(accumulator * multiplier))
```

Saturation clamps values to `[-127, 127]` instead of allowing signed overflow.
The implementation uses a floating-point multiplier to keep this teaching
version readable. Production integer-only kernels encode the same multiplier as
a fixed-point integer plus a shift.

### Trace one inference

```bash
./build/sensor_classifier
```

The program prints the quantized input, hidden activation after ReLU, final
class scores, prediction, model size, and activation-storage size. The included
weights are deliberately small and hand-picked so each neuron can be calculated
with pencil and paper. A trained deployment would export learned int8 weights
into the same `DenseLayer` layout.

The example models vibration data because it is representative of embedded edge
AI: a microcontroller can extract a few bounded signal features, classify the
machine locally, and transmit an alert without storing raw sensor streams or
requiring a network connection.

## Build & test

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
./build/quantized_matmul
./build/sensor_classifier
```

Requires CMake ≥ 3.20 and a C++17 compiler. Tests are dependency-free.

## Roadmap

- Per-axis (per-channel) quantization and asymmetric zero-points.
- Integer-only fixed-point requantization and fused activation kernels.
- A compact, versioned model format plus a desktop weight-conversion tool.
- SIMD (NEON / AVX) and a CMSIS-NN-style path for Cortex-M, so models from this
  core can drop straight onto the `mcu-baremetal` targets.

## License

MIT — see [`LICENSE`](LICENSE).
