#include <iostream>
#include "CudaIpcMemoryManagerAPI.h"

int main(int argc, char** argv) {
  try {
    // Create an instance of the IPC memory manager API
    cuda::ipc::api::CudaIpcMemoryManagerAPI api(
        "ipc:///tmp/cuda-ipc-memory-manager-service.ipc"
        );

    // query for avaiable gpus
    auto gpus = api.GetAvailableGPUs();
    for (auto gpu : gpus) {
      // query info
      auto total_buffer_count   = api.GetAllocatedTotalBufferCount(gpu);
      auto total_bytes          = api.GetAllocatedTotalBytes(gpu);
      auto max_allocation_bytes = api.GetMaxAllocationBytes(gpu);
      std::cout << "gpu: " << gpu << " allocated buffers: " << total_buffer_count << " allocated bytes: " << total_bytes << " max allocation bytes: "
          << max_allocation_bytes << std::endl;
    }
  } catch (const std::exception& e) {
    // Print any exceptions that occur
    std::cerr << "Exception: " << e.what() << std::endl;
  }

  std::cout << "Exiting..." << std::endl;
  return 0;
}