#include <boost/uuid/uuid.hpp>            // for boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp> // for string_generator
#include <iostream>
#include "CudaIpcMemoryManagerAPI.h"

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

    // Free the GPU buffer associated with the given UUID
    api.FreeCUDABufferRequest(boost_buffer_id);

    std::cout << "GPU buffer freed successfully." << std::endl;
  } catch (const std::exception& e) {
    // Print any exceptions that occur
    std::cerr << "Exception: " << e.what() << std::endl;
  }

  std::cout << "Exiting..." << std::endl;
  return 0;
}