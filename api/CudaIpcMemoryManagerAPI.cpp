#include "CudaIpcMemoryManagerAPI.h"

// include gen fbs in implementation
#include "UUIDConverter.hpp"
#include "api/rpc_request_generated.h"
#include "api/rpc_response_generated.h"

namespace cuda::ipc::api {
CudaIpcMemoryManagerAPI::CudaIpcMemoryManagerAPI(const std::string& endpoint, int receive_timeout_ms)
  : context_(1), socket_(context_, zmq::socket_type::req) {
  socket_.connect(endpoint);
  socket_.set(zmq::sockopt::rcvtimeo, receive_timeout_ms);
  spdlog::cfg::load_env_levels();
}

CudaIpcMemoryManagerAPI::~CudaIpcMemoryManagerAPI() {
}

// Returns a list of available GPU UUIDs.
std::vector<boost::uuids::uuid> CudaIpcMemoryManagerAPI::GetAvailableGPUs() {
  // Build FlatBuffer IPC request
  flatbuffers::FlatBufferBuilder builder;
  auto                           req = fbs::cuda::ipc::api::CreateGetAvailableGPUsRequest(builder);
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_GetAvailableGPUsRequest, req.o);
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

  spdlog::debug("Received response : {}", response_msg.size());

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

  // parse GetAvailableGPUsResponse
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_GetAvailableGPUsResponse);
  auto gpus_response = rpc_response->response_as_GetAvailableGPUsResponse();
  if (!gpus_response) {
    throw std::runtime_error("Invalid GetAvailableGPUsResponse in RPC response.");
  }

  // convert fbs gpu uuids to boost uuid vector
  std::vector<boost::uuids::uuid> gpu_uuids;
  auto                            gpus = gpus_response->gpus();
  if (gpus) {
    for (auto gpu_uuid : *gpus) {
      gpu_uuids.push_back(util::UUIDConverter::toBoostUUID(*gpu_uuid));
    }
  }
  // return gpu_uuids
  return gpu_uuids;
}

// Returns the total number of allocated buffers for the specified GPU.
uint64_t CudaIpcMemoryManagerAPI::GetAllocatedTotalBufferCount(const boost::uuids::uuid& gpu_uuid) {
  // Build FlatBuffer IPC request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fbs_gpu_uuid = util::UUIDConverter::toFlatBufferUUID(gpu_uuid);
  auto                           req          = fbs::cuda::ipc::api::CreateGetAllocatedTotalBufferCountRequest(builder, &fbs_gpu_uuid);
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_GetAllocatedTotalBufferCountRequest, req.o);
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

  spdlog::debug("Received response : {}", response_msg.size());

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

  // parse GetAllocatedTotalBufferCountResponse
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_GetAllocatedTotalBufferCountResponse);
  auto total_count_response = rpc_response->response_as_GetAllocatedTotalBufferCountResponse();
  if (!total_count_response) {
    throw std::runtime_error("Invalid GetAllocatedTotalBufferCountResponse in RPC response.");
  }

  // return value
  return total_count_response->value();
}

// Returns the total allocated bytes for the specified GPU.
uint64_t CudaIpcMemoryManagerAPI::GetAllocatedTotalBytes(const boost::uuids::uuid& gpu_uuid) {
  // Build FlatBuffer IPC request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fbs_gpu_uuid = util::UUIDConverter::toFlatBufferUUID(gpu_uuid);
  auto                           req          = fbs::cuda::ipc::api::CreateGetAllocatedTotalBytesRequest(builder, &fbs_gpu_uuid);
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_GetAllocatedTotalBytesRequest, req.o);
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

  spdlog::debug("Received response : {}", response_msg.size());

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

  // parse GetAllocatedTotalBytesResponse
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_GetAllocatedTotalBytesResponse);
  auto total_bytes_response = rpc_response->response_as_GetAllocatedTotalBytesResponse();
  if (!total_bytes_response) {
    throw std::runtime_error("Invalid GetAllocatedTotalBytesResponse in RPC response.");
  }

  // return value
  return total_bytes_response->value();
}

// Returns the maximum allocation size (in bytes) for the specified GPU.
uint64_t CudaIpcMemoryManagerAPI::GetMaxAllocationBytes(const boost::uuids::uuid& gpu_uuid) {
  // Build FlatBuffer IPC request
  flatbuffers::FlatBufferBuilder builder;
  auto                           fbs_gpu_uuid = util::UUIDConverter::toFlatBufferUUID(gpu_uuid);
  auto                           req          = fbs::cuda::ipc::api::CreateGetMaxAllocationBytesRequest(builder, &fbs_gpu_uuid);
  auto msg = fbs::cuda::ipc::api::CreateRPCRequestMessage(builder, fbs::cuda::ipc::api::RPCRequest_GetMaxAllocationBytesRequest, req.o);
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

  spdlog::debug("Received response : {}", response_msg.size());

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

  // parse GetMaxAllocationBytesResponse
  assert(rpc_response->response_type() == fbs::cuda::ipc::api::RPCResponse_GetMaxAllocationBytesResponse);
  auto max_allocation_response = rpc_response->response_as_GetMaxAllocationBytesResponse();
  if (!max_allocation_response) {
    throw std::runtime_error("Invalid GetMaxAllocationBytesResponse in RPC response.");
  }

  // return value
  return max_allocation_response->value();
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

  spdlog::debug("Received response : {}", response_msg.size());

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
  auto gpu_uuid  = create_response->gpu_uuid();
  if (!gpu_uuid) {
    throw std::runtime_error("Received null gpu_uuid in response.");
  }

  auto device_id = CudaUtils::GetDeviceIdFromUUID(util::UUIDConverter::toBoostUUID(*gpu_uuid));

  // make sure response gpu buffer size matches the requested gpu buffer size
  assert(size == create_response->size());

  // set cuda device before we get memory handle
  CudaUtils::SetDevice(device_id);

  // copy fbs ipc handle to std::array
  std::array<uint8_t, 64> ipc_handle_arr;
  std::copy(ipc_handle->value()->begin(), ipc_handle->value()->end(), ipc_handle_arr.begin());

  // get device pointer from ipc_handle
  void* d_ptr = CudaUtils::OpenHandleToCudaMemory(ipc_handle_arr);

  // build return struct GPUBuffer
  return GPUBuffer(d_ptr, size, util::UUIDConverter::toBoostUUID(*buffer_id), access_id);
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

  auto access_id = get_response->access_id();
  auto size      = get_response->size();
  auto gpu_uuid  = get_response->gpu_uuid();
  if (!gpu_uuid) {
    throw std::runtime_error("Received null gpu_uuid in response.");
  }

  auto cuda_device_id = CudaUtils::GetDeviceIdFromUUID(util::UUIDConverter::toBoostUUID(*gpu_uuid));

  // set cuda device before we get memory handle
  CudaUtils::SetDevice(cuda_device_id);

  // copy fbs ipc handle to std::array
  std::array<uint8_t, 64> ipc_handle_arr;
  std::copy(ipc_handle->value()->begin(), ipc_handle->value()->end(), ipc_handle_arr.begin());

  // get device pointer from ipc_handle
  void* d_ptr = CudaUtils::OpenHandleToCudaMemory(ipc_handle_arr);

  // build return struct GPUBuffer
  return GPUBuffer(d_ptr, size, buffer_id, access_id);
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
