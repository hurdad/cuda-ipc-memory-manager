#ifndef CUDA_IPC_MEMORY_REQUEST_API_H
#define CUDA_IPC_MEMORY_REQUEST_API_H

#include <zmq.hpp>              // cpp-zmq
#include <spdlog/spdlog.h> 		  // core spdlog library
#include <spdlog/cfg/env.h> 		// for spdlog::cfg::load_env_levels()
#include <boost/uuid/uuid.hpp>  // uuid class

#include "CudaUtils.h"

namespace cuda::ipc::api {
struct GPUBuffer {
private:
  void*              d_ptr; // 8 bytes on 64-bit
  size_t             size; // 8 bytes
  boost::uuids::uuid buffer_id; // 16 bytes
  uint32_t           access_id; // 4 bytes
  // 4 bytes padding added by compiler to align to 8 bytes

public:
  // Constructor
  GPUBuffer(const boost::uuids::uuid& id, uint32_t access, void* ptr, size_t sz)
    : buffer_id(id), access_id(access), d_ptr(ptr), size(sz) {
  }

  // Getters
  void*                     getDataPtr() const { return d_ptr; }
  size_t                    getSize() const { return size; }
  const boost::uuids::uuid& getBufferId() const { return buffer_id; }
  uint32_t                  getAccessId() const { return access_id; }
};

class CudaIpcMemoryRequestAPI {
public:
  CudaIpcMemoryRequestAPI(const std::string& endpoint);
  ~CudaIpcMemoryRequestAPI();

  GPUBuffer CreateCUDABufferRequest(uint64_t size);
  GPUBuffer GetCUDABufferRequest(const boost::uuids::uuid buffer_id);
  void      NotifyDoneRequest(const GPUBuffer& buffer);

private:
  zmq::context_t context_;
  zmq::socket_t  socket_;
  uint32_t       access_id_;
};
}
#endif //CUDA_IPC_MEMORY_REQUEST_API_H