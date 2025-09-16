#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstring>


void CudaUtils::InitDevice(const int device_id) {
  cudaError_t status = cudaSetDevice(device_id);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaSetDevice failed: " << cudaGetErrorString(status) << "\n";
    throw std::runtime_error("cudaSetDevice failed");
  }

  std::cout << "[CudaUtils] Set device to " << device_id << "\n";
}

void* CudaUtils::AllocDeviceBuffer(size_t numBytes) {
  void*       d_buffer = nullptr;
  cudaError_t status   = cudaMalloc(&d_buffer, numBytes);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaMalloc failed for " << numBytes
        << " bytes: " << cudaGetErrorString(status) << "\n";
    return nullptr;
  }

  std::cout << "[CudaUtils] Allocated device buffer at " << d_buffer
      << " (" << numBytes << " bytes)\n";
  return d_buffer;
}

void CudaUtils::FreeDeviceBuffer(void* d_buffer) {
  if (!d_buffer) return;

  cudaError_t status = cudaFree(d_buffer);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaFree failed for buffer " << d_buffer
        << ": " << cudaGetErrorString(status) << "\n";
  } else {
    std::cout << "[CudaUtils] Freed device buffer at " << d_buffer << "\n";
  }
}

bool CudaUtils::CopyToDevice(void* d_buffer, const void* h_buffer, size_t numBytes) {
  if (!d_buffer || !h_buffer) {
    std::cerr << "[CudaUtils] CopyToDevice failed: nullptr argument\n";
    return false;
  }

  cudaError_t status = cudaMemcpy(d_buffer, h_buffer, numBytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaMemcpy HtoD failed: " << cudaGetErrorString(status) << "\n";
    return false;
  }

  std::cout << "[CudaUtils] Copied " << numBytes << " bytes from host to device ("
      << d_buffer << ")\n";
  return true;
}

bool CudaUtils::CopyToHost(void* h_buffer, const void* d_buffer, size_t numBytes) {
  if (!h_buffer || !d_buffer) {
    std::cerr << "[CudaUtils] CopyToHost failed: nullptr argument\n";
    return false;
  }

  cudaError_t status = cudaMemcpy(h_buffer, d_buffer, numBytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaMemcpy DtoH failed: " << cudaGetErrorString(status) << "\n";
    return false;
  }

  std::cout << "[CudaUtils] Copied " << numBytes << " bytes from device to host ("
      << d_buffer << ")\n";
  return true;
}

fbs::cuda::ipc::api::CudaIPCHandle CudaUtils::GetCudaMemoryHandle(void* d_ptr) {
  cudaIpcMemHandle_t handle;
  cudaError_t        status = cudaIpcGetMemHandle(&handle, d_ptr);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaIpcGetMemHandle failed: " << cudaGetErrorString(status) << "\n";
    throw std::runtime_error("cudaIpcGetMemHandle failed");
  }

  std::cout << "[CudaUtils] Created CUDA IPC handle for device pointer " << d_ptr << "\n";

  // Convert CUDA handle to Flatbuffers handle
  flatbuffers::span<const uint8_t, 64> fb_span(
      reinterpret_cast<const uint8_t*>(&handle), sizeof(handle)
      );
  return fbs::cuda::ipc::api::CudaIPCHandle(fb_span);
}

void* CudaUtils::OpenHandleToCudaMemory(const fbs::cuda::ipc::api::CudaIPCHandle& cuda_ipc_handle) {
  cudaIpcMemHandle_t handle;
  static_assert(sizeof(handle) <= sizeof(cuda_ipc_handle),
                "Array size is too small for cudaIpcMemHandle_t");
  std::memcpy(&handle, &cuda_ipc_handle, sizeof(handle));

  void*       d_ptr  = nullptr;
  cudaError_t status = cudaIpcOpenMemHandle(&d_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaIpcOpenMemHandle failed: " << cudaGetErrorString(status) << "\n";
    throw std::runtime_error("cudaIpcOpenMemHandle failed");
  }

  std::cout << "[CudaUtils] Opened CUDA IPC handle, device pointer: " << d_ptr << "\n";
  return d_ptr;
}

void CudaUtils::CloseHandleToCudaMemory(void* d_ptr) {
  // close ipc gpu memory
  cudaError_t status = cudaIpcCloseMemHandle(d_ptr);
  if (status != cudaSuccess) {
    std::cerr << "[CudaUtils] cudaIpcCloseMemHandle failed: " << cudaGetErrorString(status) << "\n";
    throw std::runtime_error("cudaIpcCloseMemHandle failed");
  }

  std::cout << "[CudaUtils] Close CUDA IPC handle \n";
}