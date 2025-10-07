cmake_minimum_required(VERSION 3.28.3)

# Target RTX 3070 (Ampere - Compute Capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(cuda-ipc-memory-manager LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

add_subdirectory(schema)
add_subdirectory(util)
add_subdirectory(api)
add_subdirectory(service)
add_subdirectory(example)