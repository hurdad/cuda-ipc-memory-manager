#include <cuda_runtime.h>

#include <boost/uuid/uuid.hpp>            // for boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp> // for string_generator
#include <iostream>

#include "CudaIpcMemoryRequestAPI.h"

__global__ void incrementKernel(float* d, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d[idx] += 1.0f;
  }
}

int main(int argc, char** argv) {
  cuda::ipc::api::CudaIpcMemoryRequestAPI api("ipc:///tmp/cuda-ipc-memory-manager-service.ipc");
  boost::uuids::string_generator          gen;

  // Parse the string into a boost::uuids::uuid
  boost::uuids::uuid buffer_id  = gen("bf0b566f-f5b5-14b0-3868-eff252ba4301");
  auto               gpu_buffer = api.GetCUDABufferRequest(buffer_id);

  std::cout << "size : " << gpu_buffer.getSize() << std::endl;
  std::cout << "access_id : " << gpu_buffer.getAccessId() << std::endl;

  // access device pointer as float
  float* d_ptr = (float*)gpu_buffer.getDataPtr();

  // get buffer size
  size_t N     = gpu_buffer.getSize() / sizeof(float);

  // Do something with the data
  int threadsPerBlock = 256;
  int blocks          = (N + threadsPerBlock - 1) / threadsPerBlock;
  incrementKernel<<<blocks, threadsPerBlock>>>(d_ptr, N);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  // done with buffer
  api.NotifyDoneRequest(gpu_buffer);

  std::cout << "Exiting... " << std::endl;
  return 0;
}