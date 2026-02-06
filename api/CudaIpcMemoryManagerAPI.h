#ifndef CUDA_IPC_MEMORY_REQUEST_API_H
#define CUDA_IPC_MEMORY_REQUEST_API_H

#include <vector>
#include <array>
#include <stdexcept>

#include <spdlog/cfg/env.h> // for spdlog::cfg::load_env_levels()
#include <spdlog/spdlog.h>  // spdlog core logging library

#include <boost/uuid/uuid.hpp>    // Boost UUID class for unique buffer identification
#include <boost/uuid/uuid_io.hpp> // to_string support for UUIDs
#include <zmq.hpp>                // ZeroMQ (cpp-zmq) for inter-process communication

#include "CudaUtils.h" // Custom CUDA utility functions (user-defined)

namespace cuda::ipc::api {
// ============================================================
// GPUBuffer
// ------------------------------------------------------------
// A lightweight structure representing GPU memory allocation
// managed via CUDA IPC.
// ============================================================
struct GPUBuffer {
private:
  void*              d_ptr; // Raw pointer to allocated GPU memory (device pointer)
  size_t             size; // Total size of the GPU buffer in bytes
  boost::uuids::uuid buffer_id; // Unique identifier for this buffer
  uint32_t           access_id; // Access count or handle identifier

public:
  // Constructor
  GPUBuffer(void* ptr, size_t sz, const boost::uuids::uuid& id, uint32_t access)
    : d_ptr(ptr), size(sz), buffer_id(id), access_id(access) {
  }

  // Getters
  void*                     getDataPtr() const { return d_ptr; }
  size_t                    getSize() const { return size; }
  const boost::uuids::uuid& getBufferId() const { return buffer_id; }
  uint32_t                  getAccessId() const { return access_id; }
};

// ============================================================
// CudaIpcMemoryManagerAPI
// ------------------------------------------------------------
// This class manages CUDA IPC memory allocation requests
// and communication between different processes via ZeroMQ.
// FlatBuffers are used internally but hidden from the API.
// ============================================================
class CudaIpcMemoryManagerAPI {
public:
  // Constructor / Destructor
  explicit CudaIpcMemoryManagerAPI(const std::string& endpoint, int receive_timeout_ms = 5000);
  ~CudaIpcMemoryManagerAPI();

  // ==========================================================
  // GPU Query Methods
  // ==========================================================
  std::vector<boost::uuids::uuid> GetAvailableGPUs();
  uint64_t                        GetAllocatedTotalBufferCount(const boost::uuids::uuid& gpu_uuid);
  uint64_t                        GetAllocatedTotalBytes(const boost::uuids::uuid& gpu_uuid);
  uint64_t                        GetMaxAllocationBytes(const boost::uuids::uuid& gpu_uuid);

  // ==========================================================
  // GPU Buffer Management Methods
  // ==========================================================
  GPUBuffer CreateCUDABufferRequest(uint64_t size, boost::uuids::uuid            gpu_uuid,
                                    int32_t  expiration_access_count = 0, size_t expiration_ttl = 0,
                                    bool     zero_buffer             = false);

  GPUBuffer GetCUDABufferRequest(const boost::uuids::uuid buffer_id);
  void      NotifyDoneRequest(const GPUBuffer& buffer);
  void      FreeCUDABufferRequest(const boost::uuids::uuid buffer_id);

private:
  // ==========================================================
  // Internal Members
  // ==========================================================
  zmq::context_t context_; // ZeroMQ context for managing sockets
  zmq::socket_t  socket_; // ZeroMQ socket for sending/receiving IPC messages
};
} // namespace cuda::ipc::api

#endif // CUDA_IPC_MEMORY_REQUEST_API_H
