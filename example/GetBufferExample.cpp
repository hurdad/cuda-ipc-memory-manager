#include <iostream>
#include <boost/uuid/uuid.hpp>             // for boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>  // for string_generator

#include "CudaIpcMemoryRequestAPI.h"

int main(int argc, char** argv) {
  cuda::ipc::api::CudaIpcMemoryRequestAPI api("ipc:///tmp/cuda-ipc-memory-manager.sock");
  boost::uuids::string_generator          gen;

  // Parse the string into a boost::uuids::uuid
  boost::uuids::uuid buffer_id  = gen("550e8400-e29b-41d4-a716-446655440000");
  auto               gpu_buffer = api.GetCUDABufferRequest(buffer_id);

  void* d_ptr = gpu_buffer.getDataPtr();
  // Do something with the data

  // done with buffer
  api.NotifyDoneRequest(gpu_buffer);

  std::cout << "Exiting... " << std::endl;
  return 0;
}