#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "api/common_generated.h"
#include <iostream>

struct CudaUtils {
  // Allocate device buffer (in bytes)
  static void* AllocDeviceBuffer(size_t numBytes);

  // Free device buffer
  static void FreeDeviceBuffer(void* d_buffer);

  // Copy data from host to device
  static bool CopyToDevice(void* d_buffer, const void* h_buffer, size_t numBytes);

  // Copy data from device to host
  static bool CopyToHost(void* h_buffer, const void* d_buffer, size_t numBytes);

  // Convert device buffer => handle
  static fbs::cuda::ipc::api::CudaIPCHandle GetCudaMemoryHandle(void* d_ptr);

  // Convert handle => device buffer
  static void* HandleToCudaMemory(const fbs::cuda::ipc::api::CudaIPCHandle& cuda_ipc_handle);
};

#endif // CUDA_UTILS_H