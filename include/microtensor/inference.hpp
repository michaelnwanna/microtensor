// SPDX-License-Identifier: MIT
//
// inference.hpp - A minimal, statically sized int8 inference pipeline.
//
// This deliberately implements only dense layers, ReLU, and ArgMax. That small
// vocabulary is enough for useful classifiers while keeping every multiply,
// buffer, scale conversion, and byte of storage visible to the reader.

#ifndef MT_INFERENCE_HPP
#define MT_INFERENCE_HPP

#include <array>
#include <cstddef>
#include <cstdint>

#include "microtensor/matrix.hpp"
#include "microtensor/quantize.hpp"

namespace mt {

/// A fully connected int8 layer: output = requantize(weights * input + bias).
///
/// Bias is int32 because it is added to the dot-product accumulator. A bias
/// trained in real units is quantized with inputScale * weightScale. The
/// requantization multiplier is inputScale * weightScale / outputScale.
template <std::size_t Inputs, std::size_t Outputs>
struct DenseLayer {
    Matrix<int8_t, Outputs, Inputs> weights{};
    std::array<int32_t, Outputs> bias{};
    float requantMultiplier = 1.0f;

    std::array<int8_t, Outputs> operator()(
        const std::array<int8_t, Inputs>& input) const noexcept {
        std::array<int8_t, Outputs> output{};
        for (std::size_t row = 0; row < Outputs; ++row) {
            int32_t accumulator = bias[row];
            for (std::size_t column = 0; column < Inputs; ++column) {
                accumulator += static_cast<int32_t>(weights(row, column)) *
                               static_cast<int32_t>(input[column]);
            }
            output[row] = requantize(accumulator, requantMultiplier);
        }
        return output;
    }
};

template <std::size_t Size>
void relu(std::array<int8_t, Size>& values) noexcept {
    for (int8_t& value : values) {
        if (value < 0) {
            value = 0;
        }
    }
}

template <std::size_t Size>
std::size_t argmax(const std::array<int8_t, Size>& values) noexcept {
    static_assert(Size > 0, "argmax requires at least one value");
    std::size_t best = 0;
    for (std::size_t index = 1; index < Size; ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return best;
}

/// Two dense layers form a tiny multilayer perceptron:
/// input -> dense -> ReLU -> dense -> ArgMax.
///
/// Dimensions are template arguments, so invalid layer connections fail at
/// compile time and all intermediate storage has a fixed, inspectable size.
template <std::size_t Inputs, std::size_t Hidden, std::size_t Outputs>
struct SequentialClassifier {
    struct Result {
        std::array<int8_t, Hidden> hidden{};
        std::array<int8_t, Outputs> scores{};
        std::size_t predictedClass = 0;
    };

    DenseLayer<Inputs, Hidden> hiddenLayer{};
    DenseLayer<Hidden, Outputs> outputLayer{};

    Result infer(const std::array<int8_t, Inputs>& input) const noexcept {
        Result result;
        result.hidden = hiddenLayer(input);
        relu(result.hidden);
        result.scores = outputLayer(result.hidden);
        result.predictedClass = argmax(result.scores);
        return result;
    }
};

}  // namespace mt

#endif  // MT_INFERENCE_HPP