// SPDX-License-Identifier: MIT
//
// quantized_matmul.cpp — Demonstrates int8 inference vs. a float reference.
//
// Builds a small linear layer (A * B), computes the exact float result, then
// quantizes both operands to int8, runs the integer kernel, and reports the
// maximum absolute error introduced by quantization.

#include <cstdio>

#include "microtensor/matrix.hpp"
#include "microtensor/quantize.hpp"

int main() {
    mt::Matrix<float, 2, 3> a;
    a(0, 0) = 0.50f;  a(0, 1) = -0.25f;  a(0, 2) = 1.00f;
    a(1, 0) = -0.75f; a(1, 1) = 0.10f;   a(1, 2) = 0.40f;

    mt::Matrix<float, 3, 2> b;
    b(0, 0) = 1.00f;  b(0, 1) = 0.20f;
    b(1, 0) = -0.50f; b(1, 1) = 0.80f;
    b(2, 0) = 0.30f;  b(2, 1) = -1.00f;

    // Exact float reference.
    const mt::Matrix<float, 2, 2> ref = a.matmul(b);

    // Quantize to int8, run the integer kernel, dequantize.
    mt::Matrix<int8_t, 2, 3> qa;
    mt::Matrix<int8_t, 3, 2> qb;
    const mt::QuantParams pa = mt::quantizeMatrix(a, qa);
    const mt::QuantParams pb = mt::quantizeMatrix(b, qb);
    const mt::Matrix<float, 2, 2> out = mt::quantizedMatmul(qa, pa, qb, pb);

    float maxErr = 0.0f;
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            const float e = ref(i, j) - out(i, j);
            const float a_e = e < 0.0f ? -e : e;
            if (a_e > maxErr) {
                maxErr = a_e;
            }
            std::printf("ref[%zu][%zu]=% .4f  int8=% .4f\n", i, j,
                        static_cast<double>(ref(i, j)),
                        static_cast<double>(out(i, j)));
        }
    }
    std::printf("max abs error from int8 quantization: %.5f\n",
                static_cast<double>(maxErr));
    return 0;
}
