#include <iostream>
#include <boost/uuid/uuid_io.hpp>
#include <cuda_runtime.h>

#include "CudaIpcMemoryRequestAPI.h"

__global__ void incrementKernel(float* d, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    d[idx] += 1.0f;
  }
}

int main(int argc, char** argv) {
  cuda::ipc::api::CudaIpcMemoryRequestAPI api("ipc:///tmp/cuda-ipc-memory-manager.sock");

  // request  gpu buffer
  int N = 10;
  auto gpu_buffer = api.CreateCUDABufferRequest(N * sizeof(float));

  // get buffer id (uuid
  auto        buffer_id = gpu_buffer.getBufferId();
  std::string uuid_str  = boost::uuids::to_string(buffer_id);
  std::cout << "Buffer ID: " << uuid_str << std::endl;

  // access device pointer as float
  float* d_ptr = (float*)gpu_buffer.getDataPtr();

  // Do something with the data
  int threadsPerBlock = 256;
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  incrementKernel<<<blocks, threadsPerBlock>>>(d_ptr, N);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  // done with buffer
  api.NotifyDoneRequest(gpu_buffer);

  std::cout << "Exiting... " << std::endl;
  return 0;
}