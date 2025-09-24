#ifndef CUDA_IPC_SERVER_H
#define CUDA_IPC_SERVER_H

#include <spdlog/spdlog.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

// Prometheus Metrics
#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

// Boost UUID
#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp> // for to_string

// UUID Util
#include "UUIDConverter.hpp"

// CUDA Utils
#include "CudaUtils.h"

// FlatBuffers header generated from schema .fbs
#include "api/rpc_request_generated.h"
#include "api/rpc_response_generated.h"
#include "service_generated.h"

// GPU Metrics
#include "GpuMetricsCollector.h"

// Gpu Buffer Multi Index
#include "GPUBufferMultiIndex.h"


class CudaIPCServer {
public:
  CudaIPCServer(const fbs::cuda::ipc::service::Configuration* configuration);
  ~CudaIPCServer();

  void start();
  void stop();
  void join();

private:
  const fbs::cuda::ipc::service::Configuration* configuration_;
  zmq::context_t                                context_;
  zmq::socket_t                                 socket_;
  std::thread                                   server_thread_;
  std::thread                                   expiration_thread_;
  std::atomic<bool>                             running_; // stop flag
  GPUBufferMultiIndex                           buffers_; // container that holds GPUBufferRecord
  std::mutex                                    buffers_mutex_; // <-- Protects access to buffers_
  std::unordered_map<int32_t, size_t>           max_gpu_allocated_memory_; // max gpu allocated memory mapped by gpu_device_index

  // Main server loop
  void run();
  // Expiration thread loop
  void expirationLoop(uint32_t expiration_thread_interval_ms);
  void cleanupExpiredBuffers();

  // Functions for handling individual request types
  void handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req,
                          flatbuffers::FlatBufferBuilder&                     response_builder,
                          std::chrono::time_point<std::chrono::steady_clock>  start_timestamp);
  void handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest*   req,
                       flatbuffers::FlatBufferBuilder&                    response_builder,
                       std::chrono::time_point<std::chrono::steady_clock> start_timestamp);
  void handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest*      req,
                        flatbuffers::FlatBufferBuilder&                    response_builder,
                        std::chrono::time_point<std::chrono::steady_clock> start_timestamp);
  void handleFreeBuffer(const fbs::cuda::ipc::api::FreeCUDABufferRequest*  req,
                        flatbuffers::FlatBufferBuilder&                    response_builder,
                        std::chrono::time_point<std::chrono::steady_clock> start_timestamp);

  static boost::uuids::uuid generateUUID();
  static bool               setThreadRealtime(std::thread& t);

  // Prometheus metrics
  std::shared_ptr<prometheus::Registry> registry_;
  std::unique_ptr<prometheus::Exposer>  exposer_;
  prometheus::Counter*                  requests_total_;
  prometheus::Counter*                  errors_total_;
  prometheus::Counter*                  create_buffer_success_;
  prometheus::Counter*                  create_buffer_fail_;
  std::unordered_map<int, prometheus::Gauge*> allocated_buffers_map_;
  std::unordered_map<int, prometheus::Gauge*> allocated_bytes_map_;
  prometheus::Counter*                  expired_buffers_;
  prometheus::Histogram*                create_buffer_latency_;
  prometheus::Histogram*                get_buffer_latency_;
  prometheus::Histogram*                notify_done_latency_;
  prometheus::Histogram*                free_buffer_latency_;
  prometheus::Histogram*                expire_buffers_latency_;
};

#endif // CUDA_UTILS_H