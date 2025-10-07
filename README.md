# cuda-ipc-memory-manager
CUDA IPC Memory Manager (Service + API)

## Overview
cuda-ipc-memory-manager is a utility designed to facilitate efficient GPU memory sharing between multiple processes using CUDA's Inter-Process Communication (IPC) capabilities. It provides both a service and an API to simplify the management of IPC memory handles, making it easier to build high-performance, multi-process CUDA applications. This tool is particularly useful for deep learning, scientific computing, and distributed GPU workloads where seamless and efficient memory sharing is essential.

## Features

- **IPC Memory Management Service**: Centralized service for managing CUDA IPC memory handles across multiple processes
- **C++ API**: Easy-to-use C++ API for allocating, sharing, and managing GPU memory via IPC
- **UUID-based Handle Management**: Track and reference shared memory allocations using unique identifiers
- **CUDA Utilities**: Helper utilities for CUDA operations and error handling
- **FlatBuffers Schema**: Efficient serialization for communication between service and clients
- **Observability**: Built-in monitoring and logging capabilities
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **Comprehensive Testing**: Unit tests for core components including CUDA utilities and UUID conversion

## Key Components

- **`api/`**: Client library for interacting with the IPC memory manager service
- **`service/`**: Core service managing CUDA IPC memory handles and allocation requests
- **`schema/`**: FlatBuffers schemas for efficient serialization of messages between clients and service
- **`util/`**: Utility functions for CUDA operations and UUID conversion
- **`tests/`**: Unit tests for validating functionality
- **`observability/`**: Monitoring and logging infrastructure
- **`example/`**: Example applications demonstrating usage

## Dependencies

**Build Requirements:**
- **CMake**: 3.18 or higher (build system)
- **C++ Compiler**: C++17 compatible compiler
- **NVIDIA CUDA Toolkit**: 11.0 or higher
- **FlatBuffers**: For schema serialization/deserialization
- **Google Test**: For unit testing (optional, for running tests)

**Runtime Requirements:**
- **NVIDIA GPU**: With CUDA compute capability 3.5 or higher
- **NVIDIA Driver**: Compatible with CUDA Toolkit version
- **Docker** (optional): For containerized deployment
- **Docker Compose** (optional): For orchestrated multi-container setup

**CUDA IPC Requirements:**
- Processes must run on the same physical machine
- GPU must support CUDA IPC (most modern NVIDIA GPUs do)
- Linux operating system (CUDA IPC has limited support on other platforms)