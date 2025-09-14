#include "CudaIPCServer.h"

#include <spdlog/spdlog.h>

CudaIPCServer::CudaIPCServer(const std::string& endpoint)
  : context_(1),
    socket_(context_, zmq::socket_type::rep),
    endpoint_(endpoint),
    running_(false) {
  socket_.bind(endpoint_);
  socket_.set(zmq::sockopt::rcvtimeo, 500);

  spdlog::info("Server listening on {}", endpoint_);
}

CudaIPCServer::~CudaIPCServer() {
  if (thread_.joinable())
    thread_.join();
}

void CudaIPCServer::start() {
  running_ = true;
  thread_  = std::thread(&CudaIPCServer::run, this);
}

void CudaIPCServer::stop() {
  running_ = false;
}

void CudaIPCServer::join() {
  if (thread_.joinable())
    thread_.join();
}

void CudaIPCServer::run() {
  while (running_) {
    zmq::message_t request_msg;
    spdlog::trace("Waiting for request...");
    auto recv_result = socket_.recv(request_msg, zmq::recv_flags::none);
    if (!recv_result)
      continue;

    auto buf  = request_msg.data();
    auto size = request_msg.size();

    auto rpc_request = fbs::cuda::ipc::api::GetRPCRequestMessage(buf);
    auto req_type    = rpc_request->request_type();

    flatbuffers::FlatBufferBuilder builder;

    switch (req_type) {
      case fbs::cuda::ipc::api::RPCRequest_CreateCUDABufferRequest:
        handleCreateBuffer(rpc_request->request_as_CreateCUDABufferRequest(), builder);
        break;

      case fbs::cuda::ipc::api::RPCRequest_GetCUDABufferRequest:
        handleGetBuffer(rpc_request->request_as_GetCUDABufferRequest(), builder);
        break;

      case fbs::cuda::ipc::api::RPCRequest_NotifyDoneRequest:
        handleNotifyDone(rpc_request->request_as_NotifyDoneRequest(), builder);
        break;

      default:
        spdlog::error("Unknown request type!");
    }
  }
}

void CudaIPCServer::handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  auto        uuid = generateUUID();
  BufferEntry entry;
  entry.size       = req->size();
  entry.ipc_handle = std::vector<uint8_t>(8, 42); // dummy 8-byte handle
  buffers_[uuid]   = entry;

  auto uuid_struct   = fbs::cuda::ipc::api::UUID(0, 0);
  auto handle_struct = fbs::cuda::ipc::api::CudaIPCHandle();
  auto resp          = fbs::cuda::ipc::api::CreateCUDABufferResponseBuilder(builder);
  resp.add_buffer_id(&uuid_struct);
  resp.add_ipc_handle(&handle_struct);
  resp.add_success(true);
  auto resp_offset = resp.Finish();

  // auto msg = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder,
  //   fbs::cuda::ipc::api::RPCResponse_CreateCUDABufferResponse,
  //   resp_offset);
  // builder.Finish(msg);
  //
  // socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
}

void CudaIPCServer::handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  auto uuid = util::UUIDConverter::toBoostUUID(*req->buffer_id());
  auto it   = buffers_.find(uuid);
  if (it == buffers_.end()) {
    std::cerr << "Buffer not found: " << uuid << std::endl;
    return;
  }

  auto handle_struct = fbs::cuda::ipc::api::CudaIPCHandle();
  // auto resp          = fbs::cuda::ipc::api::CreateGetCUDABufferResponse(builder, handle_struct, it->second.size);
  // auto msg           = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse, resp);
  // builder.Finish(msg);
  //
  // socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
}

void CudaIPCServer::handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  auto uuid    = util::UUIDConverter::toBoostUUID(*req->buffer_id());
  bool success = buffers_.erase(uuid) > 0;

  // auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponse(builder, success);
  // auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp);
  // builder.Finish(msg);
  //
  // socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
}

boost::uuids::uuid CudaIPCServer::generateUUID() {
  // static ensures the generator is constructed only once
  static boost::uuids::random_generator gen;
  return gen();
}