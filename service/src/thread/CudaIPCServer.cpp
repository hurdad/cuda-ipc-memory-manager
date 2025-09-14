#include "CudaIPCServer.h"

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

    // builder for response
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

    // send reply flatbuffers binary response
    socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
  }
}

void CudaIPCServer::handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  // init flatbuffers response
  auto resp = fbs::cuda::ipc::api::CreateCUDABufferResponseBuilder(builder);

  // generate uuid
  auto uuid = generateUUID();

  // convert to uuid flatbuffer
  auto uuid_flatbuffer = util::UUIDConverter::toFlatBufferUUID(uuid);

  // save to response
  resp.add_buffer_id(&uuid_flatbuffer);
  try {
    // allocate device buffer and get handle
    auto d_ptr              = CudaUtils::AllocDeviceBuffer(req->size());
    auto cuda_memory_handle = CudaUtils::GetCudaMemoryHandle(d_ptr);

    // create entry and save device pointer, size and handle
    GPUBufferEntry entry;
    entry.d_ptr      = d_ptr;
    entry.size       = req->size();
    entry.ipc_handle = cuda_memory_handle;

    // add entry to buffers_ hash map
    buffers_[uuid] = entry;

    // save ipc handle to response
    resp.add_ipc_handle(&cuda_memory_handle);
    resp.add_success(true);
  } catch (const std::exception& e) {
    resp.add_success(false);
  }

  // finish up response message
  auto resp_offset = resp.Finish();
  auto msg         = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder,
                                                           fbs::cuda::ipc::api::RPCResponse_CreateCUDABufferResponse,
                                                           resp_offset.o);
  builder.Finish(msg);
}

void CudaIPCServer::handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  // convert flatbuffers uuid to boost uuid
  auto uuid = util::UUIDConverter::toBoostUUID(*req->buffer_id());

  // search for buffer by boost uuid
  auto it = buffers_.find(uuid);
  if (it == buffers_.end()) {
    spdlog::warn("Buffer not found");

    // return error
    auto resp = fbs::cuda::ipc::api::CreateGetCUDABufferResponseDirect(builder,
                                                                       nullptr,
                                                                       it->second.size,
                                                                       false,
                                                                       "Buffer not found");
    auto msg = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse, resp.o);
    builder.Finish(msg);
    return;
  }

  // get buffer entry in hashmap
  GPUBufferEntry gpu_buffer_entry = it->second;

  // init flatbuffers successresponse
  auto resp = fbs::cuda::ipc::api::CreateGetCUDABufferResponse(builder,
                                                               &gpu_buffer_entry.ipc_handle,
                                                               it->second.size,
                                                               true);
  auto msg = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse, resp.o);
  builder.Finish(msg);
}

void CudaIPCServer::handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  auto uuid    = util::UUIDConverter::toBoostUUID(*req->buffer_id());
  //bool success = buffers_.erase(uuid) > 0;

  auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponse(builder, true);
  auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp.o);
  builder.Finish(msg);
}

boost::uuids::uuid CudaIPCServer::generateUUID() {
  // static ensures the generator is constructed only once
  static boost::uuids::random_generator gen;
  return gen();
}