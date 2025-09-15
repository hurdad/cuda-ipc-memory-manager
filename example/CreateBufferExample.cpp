#include <iostream>
#include <boost/uuid/uuid_io.hpp>

#include "CudaIpcMemoryRequestAPI.h"

int main(int argc, char** argv) {
  cuda::ipc::api::CudaIpcMemoryRequestAPI api("ipc:///tmp/cuda-ipc-memory-manager.sock");

  auto gpu_buffer = api.CreateCUDABufferRequest(1024);

  auto        buffer_id = gpu_buffer.getBufferId();
  std::string uuid_str  = boost::uuids::to_string(buffer_id);

  std::cout << "Buffer ID: " << uuid_str << std::endl;
  void* d_ptr = gpu_buffer.getDataPtr();
  // Do something with the data

  // done with buffer
  api.NotifyDoneRequest(gpu_buffer);

  std::cout << "Exiting... " << std::endl;
  return 0;
}