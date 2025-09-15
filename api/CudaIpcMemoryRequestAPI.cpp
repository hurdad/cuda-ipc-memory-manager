#include "CudaIpcMemoryRequestAPI.h"

#include "api/rpc_request_generated.h"
#include "api/rpc_response_generated.h"

#include "UUIDConverter.hpp"

namespace cuda::ipc::api {
CudaIpcMemoryRequestAPI::CudaIpcMemoryRequestAPI(const std::string& endpoint)
  : context_(1),
    socket_(context_, zmq::socket_type::req) {
  socket_.connect(endpoint);
  spdlog::cfg::load_env_levels();
}

CudaIpcMemoryRequestAPI::~CudaIpcMemoryRequestAPI() {
}

GPUBuffer CudaIpcMemoryRequestAPI::CreateCUDABufferRequest(uint64_t size) {
  spdlog::info("Creating CUDA buffer of size {} bytes", size);
  // Build FlatBuffer request
  flatbuffers::FlatBufferBuilder builder;
  auto                           req = fbs::cuda::ipc::api::CreateCreateCUDABufferRequest(builder, size);
  auto                           msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(
      builder,
      fbs::cuda::ipc::api::RPCRequest_CreateCUDABufferRequest,
      req.o
      );
  builder.Finish(msg);

  // Send request over ZeroMQ
  socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);

  // Receive response
  zmq::message_t response_msg;
  auto           recv_result = socket_.recv(response_msg);
  if (!recv_result) {
    throw std::runtime_error("Failed to receive response from server.");
  }

  spdlog::info("Received response : {}", response_msg.size());

  // Parse response
  auto rpc_response = fbs::cuda::ipc::api::GetRPCResponseMessage(response_msg.data());
  if (!rpc_response) {
    throw std::runtime_error("Failed to parse RPC response message.");
  }

  auto create_response = rpc_response->response_as_CreateCUDABufferResponse();
  if (!create_response) {
    throw std::runtime_error("Invalid CreateCUDABufferResponse in RPC response.");
  }

  // check response success
  if (!create_response->success()) {
    throw std::runtime_error("Failed to create CUDA buffer : ");// + create_response->error()->str());
  }

  auto buffer_id = create_response->buffer_id();
  if (!buffer_id) {
    throw std::runtime_error("Received null buffer_id in response.");
  }

  auto ipc_handle = create_response->ipc_handle();
  if (!ipc_handle) {
    throw std::runtime_error("Received null ipc_handle in response.");
  }

  auto access_id = create_response->access_id();
  if (!access_id) {
    throw std::runtime_error("Received null access_id in response.");
  }

  // get device pointer from ipc_handle
  void* d_ptr = CudaUtils::HandleToCudaMemory(*ipc_handle);

  // build return struct GPUBuffer
  return GPUBuffer(util::UUIDConverter::toBoostUUID(*buffer_id), access_id, d_ptr, size);
}

GPUBuffer CudaIpcMemoryRequestAPI::GetCUDABufferRequest(const boost::uuids::uuid buffer_id) {
  //spdlog::info("Getting CUDA buffer = {}", buffer_id.str());
  // Build FlatBuffer request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fb_buffer_id = util::UUIDConverter::toFlatBufferUUID(buffer_id);
  auto                           req          = fbs::cuda::ipc::api::CreateGetCUDABufferRequest(builder, &fb_buffer_id);
  auto                           msg          = fbs::cuda::ipc::api::CreateRPCRequestMessage(
      builder,
      fbs::cuda::ipc::api::RPCRequest_GetCUDABufferRequest,
      req.o
      );
  builder.Finish(msg);

  // Send request over ZeroMQ
  socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);

  // Receive response
  zmq::message_t response_msg;
  auto           recv_result = socket_.recv(response_msg);
  if (!recv_result) {
    throw std::runtime_error("Failed to receive response from server.");
  }

  // Parse response
  auto rpc_response = fbs::cuda::ipc::api::GetRPCResponseMessage(response_msg.data());
  if (!rpc_response) {
    throw std::runtime_error("Failed to parse RPC response message.");
  }

  auto get_response = rpc_response->response_as_GetCUDABufferResponse();
  if (!get_response) {
    throw std::runtime_error("Invalid GetCUDABufferResponse in RPC response.");
  }

  // check response success
  if (!get_response->success()) {
    throw std::runtime_error("Failed to get CUDA buffer : " + get_response->error()->str());
  }

  auto ipc_handle = get_response->ipc_handle();
  if (!ipc_handle) {
    throw std::runtime_error("Received null ipc_handle in response.");
  }

  auto access_id = get_response->access_id();
  if (!access_id) {
    throw std::runtime_error("Received null access_id in response.");
  }

  auto size = get_response->size();
  if (!size) {
    throw std::runtime_error("Received null size in response.");
  }

  // get device pointer from ipc_handle
  void* d_ptr = CudaUtils::HandleToCudaMemory(*ipc_handle);

  // build return struct GPUBuffer
  return GPUBuffer(buffer_id, access_id, d_ptr, size);
}

void CudaIpcMemoryRequestAPI::NotifyDoneRequest(const GPUBuffer& gpu_buffer) {
  // Build FlatBuffer request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fb_buffer_id = util::UUIDConverter::toFlatBufferUUID(gpu_buffer.getBufferId());
  auto                           req          = fbs::cuda::ipc::api::CreateNotifyDoneRequest(builder, &fb_buffer_id, gpu_buffer.getAccessId());
  auto                           msg          = fbs::cuda::ipc::api::CreateRPCRequestMessage(
      builder,
      fbs::cuda::ipc::api::RPCRequest_NotifyDoneRequest,
      req.o
      );
  builder.Finish(msg);

  // Send request over ZeroMQ
  socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);

  // Receive response
  zmq::message_t response_msg;
  auto           recv_result = socket_.recv(response_msg);
  if (!recv_result) {
    throw std::runtime_error("Failed to receive response from server.");
  }

  // Parse response
  auto rpc_response = fbs::cuda::ipc::api::GetRPCResponseMessage(response_msg.data());
  if (!rpc_response) {
    throw std::runtime_error("Failed to parse RPC response message.");
  }

  auto notify_response = rpc_response->response_as_NotifyDoneResponse();
  if (!notify_response) {
    throw std::runtime_error("Invalid NotifyDoneResponse in RPC response.");
  }

  // check response success
  if (!notify_response->success()) {
    throw std::runtime_error("Failed to notify Done : " + notify_response->error()->str());
  }
}
}