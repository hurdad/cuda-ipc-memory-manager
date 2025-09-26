#include <cuda_runtime.h>

#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <iostream>

#include "CudaIpcMemoryManagerAPI.h"

// CUDA kernel: increment each element of an array by 1
__global__ void incrementKernel(float* d, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d[idx] += 1.0f;
  }
}

int main(int argc, char** argv) {
  // try {
  //  Create an instance of the IPC memory manager API
  cuda::ipc::api::CudaIpcMemoryManagerAPI api("ipc:///tmp/cuda-ipc-memory-manager-service.ipc");

  std::string gpu_uuid = "ab29f361-2479-f8c8-f2e2-a56c3692a3ac";

  // Parameters for GPU buffer request
  int                            N = 268435456; // Number of floats
  boost::uuids::string_generator gen;
  boost::uuids::uuid             boost_gpu_uuid      = gen(gpu_uuid); // GPU UUID
  int                            expire_access_count = 0;             // Access count before expiration (0=disable)
  int                            expire_ttl          = 0;            // Time-to-live in seconds before expiration (0=disable)
  bool                           zero_buffer         = false;          // Initialize buffer to zero

  // Request GPU buffer
  auto gpu_buffer = api.CreateCUDABufferRequest(N * sizeof(float), boost_gpu_uuid, expire_access_count, expire_ttl, zero_buffer);

  // Get buffer ID (UUID) and print it
  auto        buffer_id = gpu_buffer.getBufferId();
  std::string uuid_str  = boost::uuids::to_string(buffer_id);
  std::cout << "Buffer ID: " << uuid_str << std::endl;

  // Access the device pointer as a float array
  float* d_ptr = static_cast<float*>(gpu_buffer.getDataPtr());

  // Launch CUDA kernel to increment each element
  int threadsPerBlock = 256;
  int blocks          = (N + threadsPerBlock - 1) / threadsPerBlock;
  incrementKernel<<<blocks, threadsPerBlock>>>(d_ptr, N);

  // Wait for GPU to finish execution
  cudaDeviceSynchronize();

  // Notify the IPC manager that we are done with the buffer
  api.NotifyDoneRequest(gpu_buffer);
  // } catch (const std::exception& e) {
  //   std::cerr << "Exception: " << e.what() << std::endl;
  // }

  std::cout << "Exiting..." << std::endl;
  return 0;
}