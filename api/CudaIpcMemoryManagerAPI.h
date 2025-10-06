#ifndef CUDA_IPC_MEMORY_REQUEST_API_H
#define CUDA_IPC_MEMORY_REQUEST_API_H

// -------------------------------
// External Libraries
// -------------------------------
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
  void*              d_ptr;                 // Raw pointer to allocated GPU memory (device pointer)
  size_t             size;                  // Total size of the GPU buffer in bytes
  boost::uuids::uuid buffer_id;             // Unique identifier for this buffer
  uint32_t           access_id;             // Access count or handle identifier
  int                cuda_device_id;        // CUDA device id/index where this buffer resides

 public:
  // ------------------------------------------------------------
  // Constructor
  // ------------------------------------------------------------
  GPUBuffer(void* ptr, size_t sz, const boost::uuids::uuid& id, uint32_t access, int device_id)
      : d_ptr(ptr), size(sz), buffer_id(id), access_id(access), cuda_device_id(device_id) {
  }

  // ------------------------------------------------------------
  // Getters
  // ------------------------------------------------------------
  void* getDataPtr() const {
    return d_ptr;
  }
  size_t getSize() const {
    return size;
  }
  const boost::uuids::uuid& getBufferId() const {
    return buffer_id;
  }
  uint32_t getAccessId() const {
    return access_id;
  }
  int getCudaDeviceId() const {
    return cuda_device_id;
  }
};

// ============================================================
// CudaIpcMemoryManagerAPI
// ------------------------------------------------------------
// This class manages CUDA IPC memory allocation requests
// and communication between different processes via ZeroMQ.
// ============================================================
class CudaIpcMemoryManagerAPI {
 public:
  // ------------------------------------------------------------
  // Constructor / Destructor
  // ------------------------------------------------------------
  explicit CudaIpcMemoryManagerAPI(const std::string& endpoint);
  ~CudaIpcMemoryManagerAPI();

  // ------------------------------------------------------------
  // API Methods
  // ------------------------------------------------------------

  // Returns a list of available GPU UUIDs.
  std::vector<boost::uuids::uuid> GetAvailableGPUs();

  // Returns the total number of allocated buffers for the specified GPU.
  uint64_t GetAllocatedTotalBufferCount(const boost::uuids::uuid& gpu_uuid);

  // Returns the total allocated bytes for the specified GPU.
  uint64_t GetAllocatedTotalBytes(const boost::uuids::uuid& gpu_uuid);

  // Returns the maximum allocation size (in bytes) for the specified GPU.
  uint64_t GetMaxAllocationBytes(const boost::uuids::uuid& gpu_uuid);

  /**
   * @brief Request creation of a new CUDA buffer.
   *
   * @param size                     Size of the buffer to allocate (in bytes)
   * @param gpu_uuid                  UUID of the target GPU device (default: 0)
   * @param expiration_access_count   Access count limit before the buffer expires (default: 0, disabled)
   * @param expiration_ttl            Time-to-live before the buffer expires (default: 0, infinite)
   * @param zero_buffer               If true, zero-initialize the buffer (default: false)
   * @return GPUBuffer                Struct containing buffer metadata
   */
  GPUBuffer CreateCUDABufferRequest(uint64_t size, boost::uuids::uuid gpu_uuid, int32_t expiration_access_count = 0, size_t expiration_ttl = 0,
                                    bool zero_buffer = false);

  /**
   * @brief Retrieve an existing CUDA buffer by its unique ID.
   *
   * @param buffer_id Unique identifier of the buffer to fetch.
   * @return GPUBuffer  Struct with buffer metadata.
   */
  GPUBuffer GetCUDABufferRequest(const boost::uuids::uuid buffer_id);

  /**
   * @brief Notify the manager that the buffer usage is complete.
   *
   * @param buffer The buffer being released by the caller.
   */
  void NotifyDoneRequest(const GPUBuffer& buffer);

  /**
   * @brief Free a CUDA buffer by its unique ID.
   *
   * @param buffer_id Unique identifier of the buffer to free.
   */
  void FreeCUDABufferRequest(const boost::uuids::uuid buffer_id);

 private:
  // ------------------------------------------------------------
  // Internal Members
  // ------------------------------------------------------------
  zmq::context_t context_; // ZeroMQ context for managing sockets
  zmq::socket_t  socket_;  // ZeroMQ socket for sending/receiving IPC messages
};
} // namespace cuda::ipc::api

#endif // CUDA_IPC_MEMORY_REQUEST_API_H