// SPDX-License-Identifier: MIT
//
// quantize.hpp — Symmetric int8 quantization + an integer matmul kernel.
//
// This is the heart of on-device inference: weights/activations are stored as
// int8 with a single float scale, multiplies accumulate in int32, and only the
// final result is dequantized. The same idea underpins TFLite-Micro / CMSIS-NN
// kernels, written here from first principles with no dependencies.

#ifndef MT_QUANTIZE_HPP
#define MT_QUANTIZE_HPP

#include <cstddef>
#include <cstdint>

#include "microtensor/matrix.hpp"

namespace mt {

/// Symmetric quantization: x ≈ q * scale, with q in [-127, 127].
struct QuantParams {
    float scale = 1.0f;
};

/// Pick a symmetric scale so that max|x| maps to 127.
inline QuantParams chooseSymmetricScale(const float* x, std::size_t n) noexcept {
    float maxabs = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = x[i] < 0.0f ? -x[i] : x[i];
        if (a > maxabs) {
            maxabs = a;
        }
    }
    QuantParams p;
    p.scale = (maxabs > 0.0f) ? (maxabs / 127.0f) : 1.0f;
    return p;
}

/// Quantize one value with round-half-away-from-zero and saturation.
inline int8_t quantize(float x, const QuantParams& p) noexcept {
    const float v = x / p.scale;
    int32_t q = static_cast<int32_t>(v >= 0.0f ? v + 0.5f : v - 0.5f);
    if (q > 127) {
        q = 127;
    } else if (q < -127) {
        q = -127;
    }
    return static_cast<int8_t>(q);
}

inline float dequantize(int8_t q, const QuantParams& p) noexcept {
    return static_cast<float>(q) * p.scale;
}

/// Quantize a whole matrix in place into an int8 matrix, returning its scale.
template <std::size_t R, std::size_t C>
QuantParams quantizeMatrix(const Matrix<float, R, C>& src,
                           Matrix<int8_t, R, C>& dst) noexcept {
    const QuantParams p = chooseSymmetricScale(src.data(), R * C);
    for (std::size_t i = 0; i < R * C; ++i) {
        dst.data()[i] = quantize(src.data()[i], p);
    }
    return p;
}

/// Integer matmul: int8 inputs, int32 accumulation, dequantized float output.
/// out ≈ A * B, where qa/qb carry A's and B's scales.
template <std::size_t R, std::size_t K, std::size_t C>
Matrix<float, R, C> quantizedMatmul(const Matrix<int8_t, R, K>& a,
                                    const QuantParams& qa,
                                    const Matrix<int8_t, K, C>& b,
                                    const QuantParams& qb) noexcept {
    Matrix<float, R, C> out{};
    const float outScale = qa.scale * qb.scale;
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = 0; j < C; ++j) {
            int32_t acc = 0;
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<int32_t>(a(i, k)) *
                       static_cast<int32_t>(b(k, j));
            }
            out(i, j) = static_cast<float>(acc) * outScale;
        }
    }
    return out;
}

}  // namespace mt

#endif  // MT_QUANTIZE_HPP
