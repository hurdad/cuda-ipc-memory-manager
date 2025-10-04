#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

#include "CudaUtils.h"

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
    std::cout << "[CudaUtils] Zero-initialized device buffer at " << d_buffer << " (" << numBytes << " bytes)\n";
  } else {
    std::cout << "[CudaUtils] Allocated device buffer at " << d_buffer << " (" << numBytes << " bytes)\n";
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
  if (!d_buffer || !h_buffer) throw std::runtime_error("[CudaUtils] CopyToDevice failed: nullptr argument");
  // Copy from host to device; throws on failure
  CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, numBytes, cudaMemcpyHostToDevice));
  std::cout << "[CudaUtils] Copied " << numBytes << " bytes from host to device (" << d_buffer << ")\n";
}

void CudaUtils::CopyToHost(void* h_buffer, const void* d_buffer, size_t numBytes) {
  if (!h_buffer || !d_buffer) throw std::runtime_error("[CudaUtils] CopyToHost failed: nullptr argument");
  // Copy from device to host; throws on failure
  CUDA_CHECK(cudaMemcpy(h_buffer, d_buffer, numBytes, cudaMemcpyDeviceToHost));
  std::cout << "[CudaUtils] Copied " << numBytes << " bytes from device to host (" << d_buffer << ")\n";
}

const std::array<uint8_t, 64> CudaUtils::GetCudaMemoryHandle(void* d_ptr) {
  cudaIpcMemHandle_t handle;
  // Create IPC handle for device memory; throws on failure
  CUDA_CHECK(cudaIpcGetMemHandle(&handle, d_ptr));

  std::cout << "[CudaUtils] Created CUDA IPC handle for device pointer " << d_ptr << "\n";

  // Copy CUDA handle into std::array
  std::array<uint8_t, 64> handle_storage;
  std::memcpy(handle_storage.data(), &handle, sizeof(handle));
  return handle_storage;
}

void* CudaUtils::OpenHandleToCudaMemory(const std::array<uint8_t, 64>& cuda_ipc_handle) {
  cudaIpcMemHandle_t handle;

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

void CudaUtils::GetMemoryInfo(size_t* free, size_t* total) {
  // Get cuda mem info; throws on failure
  CUDA_CHECK(cudaMemGetInfo(free, total));
}

int CudaUtils::GetDeviceIdFromUUID(const boost::uuids::uuid& gpu_uuid) {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  // search devices for uuid
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

    boost::uuids::uuid device_uuid;
    std::copy(std::begin(prop.uuid.bytes), std::end(prop.uuid.bytes), device_uuid.begin());

    if (device_uuid == gpu_uuid) {
      return i; // Found matching device
    }
  }

  throw std::runtime_error("No CUDA device found with the given GPU UUID : " + boost::uuids::to_string(gpu_uuid));
}