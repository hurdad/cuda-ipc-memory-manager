#pragma once

#include <iostream>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
#include <zmq.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/functional/hash.hpp>

#include "UUIDConverter.hpp"

// FlatBuffers header generated from schema .fbs
#include "api/rpc_request_generated.h"
#include "api/rpc_response_generated.h"

struct BufferEntry {
  std::vector<uint8_t> ipc_handle;
  uint64_t             size;
};

class CudaIPCServer {
public:
  CudaIPCServer(const std::string& endpoint);
  ~CudaIPCServer();

  void start();
  void stop();
  void join();

private:
  zmq::context_t                               context_;
  zmq::socket_t                                socket_;
  std::thread                                  thread_;
  std::string                                  endpoint_;
  std::atomic<bool> running_;  // stop flag
  std::unordered_map<boost::uuids::uuid, BufferEntry, boost::hash<boost::uuids::uuid>> buffers_;

  void run();

  // Functions for handling individual request types
  void handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder);
  void handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder);
  void handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest* req, flatbuffers::FlatBufferBuilder& builder);

  boost::uuids::uuid generateUUID();
};