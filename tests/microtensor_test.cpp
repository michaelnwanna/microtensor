// SPDX-License-Identifier: MIT
//
// Dependency-free unit tests for microtensor.

#include <cstdio>

#include "microtensor/matrix.hpp"
#include "microtensor/quantize.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

float absf(float x) {
    return x < 0.0f ? -x : x;
}

void test_matmul_known_values() {
    mt::Matrix<float, 2, 2> a;
    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(1, 0) = 3.0f;
    a(1, 1) = 4.0f;

    mt::Matrix<float, 2, 2> b;
    b(0, 0) = 5.0f;
    b(0, 1) = 6.0f;
    b(1, 0) = 7.0f;
    b(1, 1) = 8.0f;

    const mt::Matrix<float, 2, 2> c = a.matmul(b);
    CHECK(absf(c(0, 0) - 19.0f) < 1e-5f);
    CHECK(absf(c(0, 1) - 22.0f) < 1e-5f);
    CHECK(absf(c(1, 0) - 43.0f) < 1e-5f);
    CHECK(absf(c(1, 1) - 50.0f) < 1e-5f);
}

void test_transpose() {
    mt::Matrix<float, 2, 3> a;
    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(0, 2) = 3.0f;
    a(1, 0) = 4.0f;
    a(1, 1) = 5.0f;
    a(1, 2) = 6.0f;

    const mt::Matrix<float, 3, 2> t = a.transpose();
    CHECK(absf(t(0, 0) - 1.0f) < 1e-5f);
    CHECK(absf(t(2, 1) - 6.0f) < 1e-5f);
    CHECK(absf(t(1, 0) - 2.0f) < 1e-5f);
}

void test_quantize_roundtrip() {
    const float xs[5] = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f};
    const mt::QuantParams p = mt::chooseSymmetricScale(xs, 5);
    for (float x : xs) {
        const int8_t q = mt::quantize(x, p);
        const float r = mt::dequantize(q, p);
        // Error is bounded by half a quantization step.
        CHECK(absf(r - x) <= p.scale);
    }
}

void test_quantized_matmul_matches_float() {
    mt::Matrix<float, 2, 3> a;
    a(0, 0) = 0.5f;
    a(0, 1) = -0.25f;
    a(0, 2) = 1.0f;
    a(1, 0) = -0.75f;
    a(1, 1) = 0.1f;
    a(1, 2) = 0.4f;

    mt::Matrix<float, 3, 2> b;
    b(0, 0) = 1.0f;
    b(0, 1) = 0.2f;
    b(1, 0) = -0.5f;
    b(1, 1) = 0.8f;
    b(2, 0) = 0.3f;
    b(2, 1) = -1.0f;

    const mt::Matrix<float, 2, 2> ref = a.matmul(b);

    mt::Matrix<int8_t, 2, 3> qa;
    mt::Matrix<int8_t, 3, 2> qb;
    const mt::QuantParams pa = mt::quantizeMatrix(a, qa);
    const mt::QuantParams pb = mt::quantizeMatrix(b, qb);
    const mt::Matrix<float, 2, 2> out = mt::quantizedMatmul(qa, pa, qb, pb);

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            CHECK(absf(ref(i, j) - out(i, j)) < 0.05f);
        }
    }
}

}  // namespace

int main() {
    test_matmul_known_values();
    test_transpose();
    test_quantize_roundtrip();
    test_quantized_matmul_matches_float();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
