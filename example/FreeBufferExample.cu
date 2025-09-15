#include <boost/uuid/uuid.hpp>            // for boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp> // for string_generator
#include <iostream>

#include "CudaIpcMemoryRequestAPI.h"

int main(int argc, char** argv) {
  cuda::ipc::api::CudaIpcMemoryRequestAPI api("ipc:///tmp/cuda-ipc-memory-manager-service.ipc");
  boost::uuids::string_generator          gen;

  // Parse the string into a boost::uuids::uuid
  boost::uuids::uuid buffer_id  = gen("bf0b566f-f5b5-14b0-3868-eff252ba4301");

  // free cuda buffer
  api.FreeCUDABufferRequest(buffer_id);

  std::cout << "Exiting... " << std::endl;
  return 0;
}