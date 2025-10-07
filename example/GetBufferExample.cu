#include <cuda_runtime.h>
#include <boost/uuid/uuid.hpp>            // for boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp> // for string_generator
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
  if (argc < 2) {
    printf("Usage: %s <buffer_id>\n", argv[0]);
    return 1;
  }

  // get buffer_id from command ine
  char* buffer_id = argv[1];

  try {
    // Create an instance of the IPC memory manager API
    cuda::ipc::api::CudaIpcMemoryManagerAPI api(
        "ipc:///tmp/cuda-ipc-memory-manager-service.ipc"
        );

    // Generator to parse UUID strings
    boost::uuids::string_generator gen;

    // Parse the string into a boost::uuids::uuid
    boost::uuids::uuid boost_buffer_id = gen(buffer_id);

    // Retrieve GPU buffer using the UUID
    auto gpu_buffer = api.GetCUDABufferRequest(boost_buffer_id);

    // Print buffer information
    std::cout << "Buffer size (bytes): " << gpu_buffer.getSize() << std::endl;
    std::cout << "Access ID: " << gpu_buffer.getAccessId() << std::endl;

    // Access the device pointer as a float array
    float* d_ptr = static_cast<float*>(gpu_buffer.getDataPtr());

    // Determine number of floats in the buffer
    size_t N = gpu_buffer.getSize() / sizeof(float);

    // Launch CUDA kernel to increment each element
    int threadsPerBlock = 256;
    int blocks          = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocks, threadsPerBlock>>>(d_ptr, N);

    // Wait for GPU to finish execution
    cudaDeviceSynchronize();

    // Copy a portion of the device buffer to host for printing
    int                printCount = 10; // number of elements to print
    std::vector<float> h_data(printCount);

    // Copy data from device to host
    cudaMemcpy(h_data.data(), d_ptr, printCount * sizeof(float), cudaMemcpyDeviceToHost);

    // Notify the IPC manager that we are done with the buffer
    api.NotifyDoneRequest(gpu_buffer);

    // Print the first few values
    std::cout << "First " << printCount << " values of GPU buffer after increment:" << std::endl;
    for (int i = 0; i < printCount; ++i) {
      std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
  } catch (const std::exception& e) {
    // Print any exceptions that occur
    std::cerr << "Exception: " << e.what() << std::endl;
  }

  std::cout << "Exiting..." << std::endl;
  return 0;
}