#include "CudaIpcMemoryManagerAPI.h"

#include "UUIDConverter.hpp"
#include "api/rpc_request_generated.h"
#include "api/rpc_response_generated.h"

namespace cuda::ipc::api {
CudaIpcMemoryManagerAPI::CudaIpcMemoryManagerAPI(const std::string& endpoint) : context_(1), socket_(context_, zmq::socket_type::req) {
  socket_.connect(endpoint);
  spdlog::cfg::load_env_levels();
}

CudaIpcMemoryManagerAPI::~CudaIpcMemoryManagerAPI() {
}

GPUBuffer CudaIpcMemoryManagerAPI::CreateCUDABufferRequest(uint64_t size, boost::uuids::uuid gpu_uuid, int32_t expiration_access_count,
                                                           size_t expiration_ttl, bool zero_buffer) {
  spdlog::info("Creating CUDA buffer of size={} bytes on gpu_uuid={}", size, boost::uuids::to_string(gpu_uuid));
  // Build FlatBuffer IPC request
  flatbuffers::FlatBufferBuilder        builder;
  fbs::cuda::ipc::api::ExpirationOption expiration_option(expiration_access_count, expiration_ttl);
  auto                                  fbs_gpu_uuid = util::UUIDConverter::toFlatBufferUUID(gpu_uuid);
  auto req = fbs::cuda::ipc::api::CreateCreateCUDABufferRequest(builder, &fbs_gpu_uuid, size, &expiration_option, zero_buffer);
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_CreateCUDABufferRequest, req.o);
  builder.Finish(msg);

  // Send request over ZeroMQ
  auto send_result = socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
  if (!send_result.has_value()) {
    throw std::runtime_error(fmt::format("ZMQ Send failed. Error: {}", zmq_strerror(zmq_errno())));
  }

  // Receive response
  zmq::message_t response_msg;
  auto           recv_result = socket_.recv(response_msg);
  if (!recv_result) {
    throw std::runtime_error("Failed to receive response from server.");
  }

  spdlog::info("Received response : {}", response_msg.size());

  // get response buffer
  auto response_buf  = response_msg.data();
  auto response_size = response_msg.size();

  // verify flatbuffers request
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(response_buf), response_size);
  if (!fbs::cuda::ipc::api::VerifyRPCResponseMessageBuffer(verifier)) {
    throw std::runtime_error("Invalid IPC Response Message");
  }
  // Parse response
  auto rpc_response = fbs::cuda::ipc::api::GetRPCResponseMessage(response_msg.data());
  if (!rpc_response) {
    throw std::runtime_error("Failed to parse RPC response message.");
  }

  // check for ErrorResponse type
  if (rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_ErrorResponse) {
    auto error_response = rpc_response->response_as_ErrorResponse();
    if (!error_response) {
      throw std::runtime_error("Invalid ErrorResponse in RPC response.");
    }
    throw std::runtime_error(error_response->message()->str());
  }

  // parse CreateCUDABufferResponse
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_CreateCUDABufferResponse);
  auto create_response = rpc_response->response_as_CreateCUDABufferResponse();
  if (!create_response) {
    throw std::runtime_error("Invalid CreateCUDABufferResponse in RPC response.");
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
  auto device_id = create_response->cuda_device_id();

  // make sure response gpu buffer size matches the requested gpu buffer size
  assert(size == create_response->size());

  // set cuda device before we get memory handle
  CudaUtils::SetDevice(device_id);

  // get device pointer from ipc_handle
  void* d_ptr = CudaUtils::OpenHandleToCudaMemory(*ipc_handle);

  // build return struct GPUBuffer
  return GPUBuffer(d_ptr, size, util::UUIDConverter::toBoostUUID(*buffer_id), access_id, device_id);
}

GPUBuffer CudaIpcMemoryManagerAPI::GetCUDABufferRequest(const boost::uuids::uuid buffer_id) {
  spdlog::info("Getting CUDA buffer = {}", boost::uuids::to_string(buffer_id));
  //  Build FlatBuffer IPC request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fb_buffer_id = util::UUIDConverter::toFlatBufferUUID(buffer_id);
  auto                           req          = fbs::cuda::ipc::api::CreateGetCUDABufferRequest(builder, &fb_buffer_id);
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_GetCUDABufferRequest, req.o);
  builder.Finish(msg);

  // Send request over ZeroMQ
  auto send_result = socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
  if (!send_result.has_value()) {
    throw std::runtime_error(fmt::format("ZMQ Send failed. Error: {}", zmq_strerror(zmq_errno())));
  }

  // Receive response
  zmq::message_t response_msg;
  auto           recv_result = socket_.recv(response_msg);
  if (!recv_result) {
    throw std::runtime_error("Failed to receive response from server.");
  }

  // get response buffer
  auto response_buf  = response_msg.data();
  auto response_size = response_msg.size();

  // verify flatbuffers response
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(response_buf), response_size);
  if (!fbs::cuda::ipc::api::VerifyRPCResponseMessageBuffer(verifier)) {
    throw std::runtime_error("Invalid IPC Response Message");
  }

  // Parse response
  auto rpc_response = fbs::cuda::ipc::api::GetRPCResponseMessage(response_msg.data());
  if (!rpc_response) {
    throw std::runtime_error("Failed to parse RPC response message.");
  }

  // check for ErrorResponse type
  if (rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_ErrorResponse) {
    auto error_response = rpc_response->response_as_ErrorResponse();
    if (!error_response) {
      throw std::runtime_error("Invalid ErrorResponse in RPC response.");
    }
    throw std::runtime_error(error_response->message()->str());
  }

  // parse GetCUDABufferResponse
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse);
  auto get_response = rpc_response->response_as_GetCUDABufferResponse();
  if (!get_response) {
    throw std::runtime_error("Invalid GetCUDABufferResponse in RPC response.");
  }

  auto ipc_handle = get_response->ipc_handle();
  if (!ipc_handle) {
    throw std::runtime_error("Received null ipc_handle in response.");
  }

  auto access_id      = get_response->access_id();
  auto size           = get_response->size();
  auto cuda_device_id = get_response->cuda_device_id();

  // set cuda device before we get memory handle
  CudaUtils::SetDevice(cuda_device_id);

  // get device pointer from ipc_handle
  void* d_ptr = CudaUtils::OpenHandleToCudaMemory(*ipc_handle);

  // build return struct GPUBuffer
  return GPUBuffer(d_ptr, size, buffer_id, access_id, cuda_device_id);
}

void CudaIpcMemoryManagerAPI::NotifyDoneRequest(const GPUBuffer& gpu_buffer) {
  // close the gpu memory handle must be locally
  CudaUtils::CloseHandleToCudaMemory(gpu_buffer.getDataPtr());

  // Build FlatBuffer IPC request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fb_buffer_id = util::UUIDConverter::toFlatBufferUUID(gpu_buffer.getBufferId());
  auto                           req          = fbs::cuda::ipc::api::CreateNotifyDoneRequest(builder, &fb_buffer_id, gpu_buffer.getAccessId());
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_NotifyDoneRequest, req.o);
  builder.Finish(msg);

  // Send request over ZeroMQ
  auto send_result = socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
  if (!send_result.has_value()) {
    throw std::runtime_error(fmt::format("ZMQ Send failed. Error: {}", zmq_strerror(zmq_errno())));
  }

  // Receive response
  zmq::message_t response_msg;
  auto           recv_result = socket_.recv(response_msg);
  if (!recv_result) {
    throw std::runtime_error("Failed to receive response from server.");
  }

  // get response buffer
  auto response_buf  = response_msg.data();
  auto response_size = response_msg.size();

  // verify flatbuffers response
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(response_buf), response_size);
  if (!fbs::cuda::ipc::api::VerifyRPCResponseMessageBuffer(verifier)) {
    throw std::runtime_error("Invalid IPC Response Message");
  }

  // Parse response
  auto rpc_response = fbs::cuda::ipc::api::GetRPCResponseMessage(response_msg.data());
  if (!rpc_response) {
    throw std::runtime_error("Failed to parse RPC response message.");
  }

  // check for ErrorResponse type
  if (rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_ErrorResponse) {
    auto error_response = rpc_response->response_as_ErrorResponse();
    if (!error_response) {
      throw std::runtime_error("Invalid ErrorResponse in RPC response.");
    }
    throw std::runtime_error(error_response->message()->str());
  }

  // assert type
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse);

  // response is empty on success
}

void CudaIpcMemoryManagerAPI::FreeCUDABufferRequest(const boost::uuids::uuid buffer_id) {
  spdlog::info("Free CUDA buffer = {}", boost::uuids::to_string(buffer_id));
  //  Build FlatBuffer request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fb_buffer_id = util::UUIDConverter::toFlatBufferUUID(buffer_id);
  auto                           req          = fbs::cuda::ipc::api::CreateFreeCUDABufferRequest(builder, &fb_buffer_id);
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_FreeCUDABufferRequest, req.o);
  builder.Finish(msg);

  // Send request over ZeroMQ
  auto send_result = socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
  if (!send_result.has_value()) {
    throw std::runtime_error(fmt::format("ZMQ Send failed. Error: {}", zmq_strerror(zmq_errno())));
  }

  // Receive response
  zmq::message_t response_msg;
  auto           recv_result = socket_.recv(response_msg);
  if (!recv_result) {
    throw std::runtime_error("Failed to receive response from server.");
  }

  // get response buffer
  auto response_buf  = response_msg.data();
  auto response_size = response_msg.size();

  // verify flatbuffers response
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(response_buf), response_size);
  if (!fbs::cuda::ipc::api::VerifyRPCResponseMessageBuffer(verifier)) {
    throw std::runtime_error("Invalid IPC Response Message");
  }

  // Parse response
  auto rpc_response = fbs::cuda::ipc::api::GetRPCResponseMessage(response_msg.data());
  if (!rpc_response) {
    throw std::runtime_error("Failed to parse RPC response message.");
  }

  // check for ErrorResponse type
  if (rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_ErrorResponse) {
    auto error_response = rpc_response->response_as_ErrorResponse();
    if (!error_response) {
      throw std::runtime_error("Invalid ErrorResponse in RPC response.");
    }
    throw std::runtime_error(error_response->message()->str());
  }

  // assert type
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_FreeCUDABufferResponse);

  // response is empty on success
}
} // namespace cuda::ipc::api