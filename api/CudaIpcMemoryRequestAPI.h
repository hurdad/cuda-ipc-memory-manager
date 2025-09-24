#ifndef CUDA_IPC_MEMORY_REQUEST_API_H
#define CUDA_IPC_MEMORY_REQUEST_API_H

#include <zmq.hpp>              // cpp-zmq
#include <spdlog/spdlog.h> 		  // core spdlog library
#include <spdlog/cfg/env.h> 		// for spdlog::cfg::load_env_levels()
#include <boost/uuid/uuid.hpp>  // uuid class
#include <boost/uuid/uuid_io.hpp> // for to_string

#include "CudaUtils.h"

namespace cuda::ipc::api {
struct GPUBuffer {
private:
  void*              d_ptr; // 8 bytes on 64-bit (raw gpu pointer)
  size_t             size; // 8 bytes
  boost::uuids::uuid buffer_id; // 16 bytes
  uint32_t           access_id; // 4 bytes
  int                gpu_device_index; // 4 bytes

public:
  // Constructor
  GPUBuffer(void*                     ptr,
            size_t                    sz,
            const boost::uuids::uuid& id,
            uint32_t                  access,
            int                       device_index)
    : d_ptr(ptr),
      size(sz),
      buffer_id(id),
      access_id(access),
      gpu_device_index(device_index) {
  }

  // Getters
  void*                     getDataPtr() const { return d_ptr; }
  size_t                    getSize() const { return size; }
  const boost::uuids::uuid& getBufferId() const { return buffer_id; }
  uint32_t                  getAccessId() const { return access_id; }
  int                       getGpuDeviceIndex() const { return gpu_device_index; }
};

class CudaIpcMemoryRequestAPI {
public:
  CudaIpcMemoryRequestAPI(const std::string& endpoint);
  ~CudaIpcMemoryRequestAPI();

  GPUBuffer CreateCUDABufferRequest(uint64_t size,
                                                           int32_t  gpu_device_index = 0,
                                                           size_t   ttl              = 0,
                                                           bool     zero_buffer      = false);
  GPUBuffer GetCUDABufferRequest(const boost::uuids::uuid buffer_id);
  void      NotifyDoneRequest(const GPUBuffer& buffer);
  void      FreeCUDABufferRequest(const boost::uuids::uuid buffer_id);

private:
  zmq::context_t context_;
  zmq::socket_t  socket_;
  uint32_t       access_id_;
};
}
#endif //CUDA_IPC_MEMORY_REQUEST_API_H