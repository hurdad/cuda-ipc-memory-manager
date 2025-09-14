#include "CudaUtils.h"

#include <cuda_runtime.h>

void* CudaUtils::AllocDeviceBuffer(size_t numBytes) {
  void*       d_buffer = nullptr;
  cudaError_t status   = cudaMalloc(&d_buffer, numBytes);
  if (status != cudaSuccess) {
    std::cerr << "CUDA malloc failed: " << cudaGetErrorString(status) << "\n";
    return nullptr;
  }
  return d_buffer;
}

void CudaUtils::FreeDeviceBuffer(void* d_buffer) {
  if (d_buffer) {
    cudaError_t status = cudaFree(d_buffer);
    if (status != cudaSuccess) {
      std::cerr << "CUDA free failed: " << cudaGetErrorString(status) << "\n";
    }
  }
}

bool CudaUtils::CopyToDevice(void* d_buffer, const void* h_buffer, size_t numBytes) {
  if (!d_buffer || !h_buffer) return false;

  cudaError_t status = cudaMemcpy(d_buffer, h_buffer, numBytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::cerr << "CUDA memcpy HtoD failed: " << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

bool CudaUtils::CopyToHost(void* h_buffer, const void* d_buffer, size_t numBytes) {
  if (!h_buffer || !d_buffer) return false;

  cudaError_t status = cudaMemcpy(h_buffer, d_buffer, numBytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::cerr << "CUDA memcpy DtoH failed: " << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

fbs::cuda::ipc::api::CudaIPCHandle CudaUtils::GetCudaMemoryHandle(void* d_ptr) {
  cudaIpcMemHandle_t handle;
  cudaError_t        status = cudaIpcGetMemHandle(&handle, d_ptr);
  if (status != cudaSuccess) {
    std::cerr << "Failed to get CUDA IPC handle: " << cudaGetErrorString(status) << "\n";
    throw std::runtime_error("cudaIpcGetMemHandle failed");
  }

  // Copy raw bytes into a vector
  //std::array<int64_t, 8> result{};
  //memcpy(result.data(), &handle, sizeof(handle));

  //flatbuffers::span<const uint8_t, 64> fb_span(
  //    reinterpret_cast<const uint8_t*>(&handle)
  //);

  auto result = fbs::cuda::ipc::api::CudaIPCHandle();
  return result;
}

void* CudaUtils::HandleToCudaMemory(const fbs::cuda::ipc::api::CudaIPCHandle& cuda_ipc_handle) {
  cudaIpcMemHandle_t handle;
  static_assert(sizeof(handle) <= sizeof(cuda_ipc_handle), "Array size is too small for cudaIpcMemHandle_t");
  memcpy(&handle, &cuda_ipc_handle, sizeof(handle));

  // Open the handle to get the device pointer
  void*       d_ptr  = nullptr;
  cudaError_t status = cudaIpcOpenMemHandle(&d_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
  if (status != cudaSuccess) {
    std::cerr << "Failed to open CUDA IPC handle: " << cudaGetErrorString(status) << "\n";
    throw std::runtime_error("cudaIpcOpenMemHandle failed");
  }

  return d_ptr;
}
