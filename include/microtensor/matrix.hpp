// SPDX-License-Identifier: MIT
//
// matrix.hpp — Compile-time-sized, heap-free matrix.
//
// Dimensions are template parameters, so storage is a single std::array with no
// dynamic allocation — suitable for desktop *and* constrained (MCU-class)
// targets. Everything is constexpr-friendly.

#ifndef MT_MATRIX_HPP
#define MT_MATRIX_HPP

#include <array>
#include <cstddef>

namespace mt {

template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix {
public:
    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;

    constexpr T& operator()(std::size_t r, std::size_t c) noexcept {
        return data_[r * Cols + c];
    }
    constexpr const T& operator()(std::size_t r, std::size_t c) const noexcept {
        return data_[r * Cols + c];
    }

    constexpr T* data() noexcept { return data_.data(); }
    constexpr const T* data() const noexcept { return data_.data(); }

    /// Dense matrix multiply: (Rows x Cols) * (Cols x N) -> (Rows x N).
    /// Loop order is i-k-j for cache-friendly, contiguous inner writes.
    template <std::size_t N>
    constexpr Matrix<T, Rows, N> matmul(
        const Matrix<T, Cols, N>& rhs) const noexcept {
        Matrix<T, Rows, N> out{};
        for (std::size_t i = 0; i < Rows; ++i) {
            for (std::size_t k = 0; k < Cols; ++k) {
                const T a = (*this)(i, k);
                for (std::size_t j = 0; j < N; ++j) {
                    out(i, j) += a * rhs(k, j);
                }
            }
        }
        return out;
    }

    constexpr Matrix<T, Cols, Rows> transpose() const noexcept {
        Matrix<T, Cols, Rows> out{};
        for (std::size_t i = 0; i < Rows; ++i) {
            for (std::size_t j = 0; j < Cols; ++j) {
                out(j, i) = (*this)(i, j);
            }
        }
        return out;
    }

private:
    std::array<T, Rows * Cols> data_{};
};

}  // namespace mt

#endif  // MT_MATRIX_HPP
