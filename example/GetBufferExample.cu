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
    try {
        // Create an instance of the IPC memory manager API
        cuda::ipc::api::CudaIpcMemoryManagerAPI api(
            "ipc:///tmp/cuda-ipc-memory-manager-service.ipc"
        );

        // Generator to parse UUID strings
        boost::uuids::string_generator gen;

        // Parse the string into a boost::uuids::uuid
        boost::uuids::uuid buffer_id = gen("9421bfce-e78f-65ca-42e8-226f77c54a76");

        // Retrieve GPU buffer using the UUID
        auto gpu_buffer = api.GetCUDABufferRequest(buffer_id);

        // Print buffer information
        std::cout << "Buffer size (bytes): " << gpu_buffer.getSize() << std::endl;
        std::cout << "Access ID: " << gpu_buffer.getAccessId() << std::endl;

        // Access the device pointer as a float array
        float* d_ptr = static_cast<float*>(gpu_buffer.getDataPtr());

        // Determine number of floats in the buffer
        size_t N = gpu_buffer.getSize() / sizeof(float);

        // Launch CUDA kernel to increment each element
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        incrementKernel<<<blocks, threadsPerBlock>>>(d_ptr, N);

        // Wait for GPU to finish execution
        cudaDeviceSynchronize();

        // Notify the IPC manager that we are done with the buffer
        api.NotifyDoneRequest(gpu_buffer);
    }
    catch (const std::exception& e) {
        // Print any exceptions that occur
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::cout << "Exiting..." << std::endl;
    return 0;
}
