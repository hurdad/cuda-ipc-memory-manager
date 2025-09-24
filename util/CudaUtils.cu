#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

void CudaUtils::SetDevice(int device_id) {
  // Set the current CUDA device; throws on failure
  CUDA_CHECK(cudaSetDevice(device_id));
  std::cout << "[CudaUtils] Set device to " << device_id << "\n";
}

void* CudaUtils::AllocDeviceBuffer(size_t numBytes, bool zeroInitialize) {
  void* d_buffer = nullptr;
  // Allocate device memory; throws on failure
  CUDA_CHECK(cudaMalloc(&d_buffer, numBytes));

  // Optionally zero-initialize the buffer
  if (zeroInitialize) {
    CUDA_CHECK(cudaMemset(d_buffer, 0, numBytes));
    std::cout << "[CudaUtils] Zero-initialized device buffer at " << d_buffer
              << " (" << numBytes << " bytes)\n";
  } else {
    std::cout << "[CudaUtils] Allocated device buffer at " << d_buffer
              << " (" << numBytes << " bytes)\n";
  }

  return d_buffer;
}

void CudaUtils::FreeDeviceBuffer(void* d_buffer) {
  if (!d_buffer) return;
  // Free device memory; throws on failure
  CUDA_CHECK(cudaFree(d_buffer));
  std::cout << "[CudaUtils] Freed device buffer at " << d_buffer << "\n";
}

void CudaUtils::CopyToDevice(void* d_buffer, const void* h_buffer, size_t numBytes) {
  if (!d_buffer || !h_buffer)
    throw std::runtime_error("[CudaUtils] CopyToDevice failed: nullptr argument");
  // Copy from host to device; throws on failure
  CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, numBytes, cudaMemcpyHostToDevice));
  std::cout << "[CudaUtils] Copied " << numBytes << " bytes from host to device (" << d_buffer << ")\n";
}

void CudaUtils::CopyToHost(void* h_buffer, const void* d_buffer, size_t numBytes) {
  if (!h_buffer || !d_buffer)
    throw std::runtime_error("[CudaUtils] CopyToHost failed: nullptr argument");
  // Copy from device to host; throws on failure
  CUDA_CHECK(cudaMemcpy(h_buffer, d_buffer, numBytes, cudaMemcpyDeviceToHost));
  std::cout << "[CudaUtils] Copied " << numBytes << " bytes from device to host (" << d_buffer << ")\n";
}

fbs::cuda::ipc::api::CudaIPCHandle CudaUtils::GetCudaMemoryHandle(void* d_ptr) {
  cudaIpcMemHandle_t handle;
  // Create IPC handle for device memory; throws on failure
  CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_ptr));

  std::cout << "[CudaUtils] Created CUDA IPC handle for device pointer " << d_ptr << "\n";

  // Convert CUDA handle to Flatbuffers span
  flatbuffers::span<const uint8_t, 64> fb_span(
      reinterpret_cast<const uint8_t*>(&handle), sizeof(handle)
      );
  return fbs::cuda::ipc::api::CudaIPCHandle(fb_span);
}

void* CudaUtils::OpenHandleToCudaMemory(const fbs::cuda::ipc::api::CudaIPCHandle& cuda_ipc_handle) {
  cudaIpcMemHandle_t handle;
  static_assert(sizeof(handle) <= sizeof(cuda_ipc_handle),
                "Array size is too small for cudaIpcMemHandle_t");

  // Copy Flatbuffers data to CUDA handle
  std::memcpy(&handle, &cuda_ipc_handle, sizeof(handle));

  void* d_ptr = nullptr;
  // Open IPC handle; throws on failure
  CUDA_CHECK(cudaIpcOpenMemHandle(&d_ptr, handle, cudaIpcMemLazyEnablePeerAccess));

  std::cout << "[CudaUtils] Opened CUDA IPC handle, device pointer: " << d_ptr << "\n";
  return d_ptr;
}

void CudaUtils::CloseHandleToCudaMemory(void* d_ptr) {
  // Close IPC handle; throws on failure
  CUDA_CHECK(cudaIpcCloseMemHandle(d_ptr));
  std::cout << "[CudaUtils] Closed CUDA IPC handle\n";
}