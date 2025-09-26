#include <boost/uuid/uuid.hpp>            // for boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp> // for string_generator
#include <iostream>
#include "CudaIpcMemoryManagerAPI.h"

int main(int argc, char** argv) {
  try {
    // Create an instance of the IPC memory manager API
    cuda::ipc::api::CudaIpcMemoryManagerAPI api(
        "ipc:///tmp/cuda-ipc-memory-manager-service.ipc"
    );

    // Generator to parse UUID strings
    boost::uuids::string_generator gen;

    // Parse the string into a boost::uuids::uuid
    boost::uuids::uuid buffer_id = gen("ab6910de-03c6-43e3-8faf-ba42651b5207");

    // Free the GPU buffer associated with the given UUID
    api.FreeCUDABufferRequest(buffer_id);

    std::cout << "GPU buffer freed successfully." << std::endl;
  }
  catch (const std::exception& e) {
    // Print any exceptions that occur
    std::cerr << "Exception: " << e.what() << std::endl;
  }

  std::cout << "Exiting..." << std::endl;
  return 0;
}
