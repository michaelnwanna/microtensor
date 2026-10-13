# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)

project(microtensor
  VERSION 0.1.0
  DESCRIPTION "A tiny, heap-free C++ tensor and int8 quantized-inference core"
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(MT_BUILD_TESTING "Build tests" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# Header-only core library.
add_library(microtensor INTERFACE)
add_library(mt::microtensor ALIAS microtensor)
target_include_directories(microtensor INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(microtensor INTERFACE cxx_std_17)

# Example: quantized matmul vs. a float reference.
add_executable(quantized_matmul examples/quantized_matmul.cpp)
target_link_libraries(quantized_matmul PRIVATE mt::microtensor)
mt_set_warnings(quantized_matmul)

if(MT_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
