// SPDX-License-Identifier: MIT
//
// sensor_classifier.cpp - A complete heap-free int8 inference walkthrough.
//
// Imagine four vibration features sampled from a small motor. A two-layer
// network classifies the machine as normal, imbalanced, or experiencing impact.
// The weights are hand-picked so the arithmetic is easy to follow; a real
// deployment would export learned int8 weights in exactly the same layout.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "microtensor/inference.hpp"

namespace {

constexpr std::array<const char*, 4> kFeatureNames = {
    "low-band energy", "high-band energy", "peak amplitude", "crest factor"};
constexpr std::array<const char*, 3> kClassNames = {"normal", "imbalance",
                                                    "impact"};

mt::SequentialClassifier<4, 5, 3> makeModel() {
    mt::SequentialClassifier<4, 5, 3> model;

    const int8_t hiddenWeights[5][4] = {
        {2, 1, -1, 0}, {-1, 2, 0, 1},  {0, -1, 2, 1},
        {1, 0, 1, -2}, {-1, -1, 1, 2},
    };
    for (std::size_t row = 0; row < 5; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            model.hiddenLayer.weights(row, column) = hiddenWeights[row][column];
        }
    }
    model.hiddenLayer.requantMultiplier = 0.25f;

    const int8_t outputWeights[3][5] = {
        {2, 1, 0, 1, 0},
        {0, 2, 1, -1, 0},
        {-1, 0, 2, 0, 2},
    };
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 5; ++column) {
            model.outputLayer.weights(row, column) = outputWeights[row][column];
        }
    }
    model.outputLayer.requantMultiplier = 0.5f;
    return model;
}

template <std::size_t Size>
void printVector(const char* label, const std::array<int8_t, Size>& values) {
    std::printf("%-18s [", label);
    for (std::size_t index = 0; index < Size; ++index) {
        std::printf("%s%d", index == 0 ? "" : ", ",
                    static_cast<int>(values[index]));
    }
    std::printf("]\n");
}

}  // namespace

int main() {
    // These are already-quantized sensor features. For example, with an input
    // scale of 0.01, the stored value 40 represents the real value 0.40.
    const std::array<int8_t, 4> input = {40, 10, 70, 50};
    const auto model = makeModel();
    const auto result = model.infer(input);

    std::printf("microtensor: heap-free vibration classifier\n\n");
    std::printf(
        "Architecture: int8[4] -> Dense[5] -> ReLU -> Dense[3] -> "
        "ArgMax\n");
    std::printf(
        "Arithmetic:   int8 multiply, int32 accumulate, int8 "
        "requantize\n\n");

    for (std::size_t index = 0; index < input.size(); ++index) {
        std::printf("  feature %zu: %-18s = %d\n", index, kFeatureNames[index],
                    static_cast<int>(input[index]));
    }
    std::printf("\n");
    printVector("quantized input", input);
    printVector("hidden after ReLU", result.hidden);
    printVector("class scores", result.scores);

    std::printf("\nprediction: class %zu (%s)\n", result.predictedClass,
                kClassNames[result.predictedClass]);
    std::printf("model storage: %zu bytes; activation storage: %zu bytes\n",
                sizeof(model), sizeof(result));
    return 0;
}