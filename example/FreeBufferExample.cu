#include <boost/uuid/uuid.hpp>            // for boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp> // for string_generator
#include <iostream>

#include "CudaIpcMemoryRequestAPI.h"

int main(int argc, char** argv) {
  cuda::ipc::api::CudaIpcMemoryRequestAPI api("ipc:///tmp/cuda-ipc-memory-manager-service.ipc");
  boost::uuids::string_generator          gen;

  // Parse the string into a boost::uuids::uuid
  boost::uuids::uuid buffer_id  = gen("9ed7ae96-10d4-64ab-37e9-b2c784b94c60");

  // free cuda buffer
  api.FreeCUDABufferRequest(buffer_id);

  std::cout << "Exiting... " << std::endl;
  return 0;
}