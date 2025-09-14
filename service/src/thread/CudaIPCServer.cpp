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
  if (server_thread_.joinable())
    server_thread_.join();
  if (expiration_thread_.joinable())
    expiration_thread_.join();
}

void CudaIPCServer::start() {
  running_       = true;
  server_thread_ = std::thread(&CudaIPCServer::run, this);

  // Start expiration cleanup thread
  expiration_thread_ = std::thread(&CudaIPCServer::expirationLoop, this);
}

void CudaIPCServer::stop() {
  running_ = false;
}

void CudaIPCServer::join() {
  if (server_thread_.joinable())
    server_thread_.join();
  if (expiration_thread_.joinable())
    expiration_thread_.join();
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
  std::lock_guard<std::mutex> lock(buffers_mutex_);

  // init flatbuffers response
  auto resp = fbs::cuda::ipc::api::CreateCUDABufferResponseBuilder(builder);

  // generate ids
  auto buffer_id = generateUUID();
  auto access_id = rand();

  // convert to uuid flatbuffer
  auto uuid_flatbuffer = util::UUIDConverter::toFlatBufferUUID(buffer_id);

  // save to response
  resp.add_buffer_id(&uuid_flatbuffer);
  try {
    // allocate device buffer and get handle
    auto d_ptr              = CudaUtils::AllocDeviceBuffer(req->size());
    auto cuda_memory_handle = CudaUtils::GetCudaMemoryHandle(d_ptr);

    // create entry and save device pointer, size and handle
    GPUBufferRecord entry;
    entry.uuid       = buffer_id;
    entry.d_ptr      = d_ptr;
    entry.size       = req->size();
    entry.ipc_handle = cuda_memory_handle;
    entry.access_ids.push_back(access_id);
    entry.creation_timestamp      = std::chrono::steady_clock::now();
    entry.last_activity_timestamp = std::chrono::steady_clock::now();

    if (req->expiration()->option_type() == fbs::cuda::ipc::api::ExpirationOptions_TtlCreationOption) {
      entry.expiration_timestamp = std::chrono::steady_clock::now() + std::chrono::seconds(req->expiration()->option_as_TtlCreationOption()->ttl());
    }

    if (req->expiration()->option_type() == fbs::cuda::ipc::api::ExpirationOptions_AccessCountOption) {
      entry.access_counter = req->expiration()->option_as_AccessCountOption()->aceess_count();
    }

    // add entry to buffers_ hash map
    buffers_[buffer_id] = entry;

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
  std::lock_guard<std::mutex> lock(buffers_mutex_);

  // convert flatbuffers uuid to boost uuid
  auto buffer_id = util::UUIDConverter::toBoostUUID(*req->buffer_id());

  // search for buffer by buffer_id uuid
  auto it = buffers_.find(buffer_id);
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
  GPUBufferRecord gpu_buffer_entry = it->second;
  gpu_buffer_entry.access_ids.push_back(req->access_id()); // save access id
  gpu_buffer_entry.last_activity_timestamp = std::chrono::steady_clock::now(); // update last activity timestamp

  // init flatbuffers success response
  auto resp = fbs::cuda::ipc::api::CreateGetCUDABufferResponse(builder,
                                                               &gpu_buffer_entry.ipc_handle,
                                                               it->second.size,
                                                               true);
  auto msg = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse, resp.o);
  builder.Finish(msg);
}

void CudaIPCServer::handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  std::lock_guard<std::mutex> lock(buffers_mutex_);

  // convert flatbuffers uuid to boost uuid
  auto buffer_id = util::UUIDConverter::toBoostUUID(*req->buffer_id());

  // search for buffer by boost uuid
  auto it = buffers_.find(buffer_id);
  if (it == buffers_.end()) {
    spdlog::warn("Buffer not found");

    auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponseDirect(builder, false, "Buffer not found");
    auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp.o);
    builder.Finish(msg);
    return;
  }

  // get buffer entry in hashmap
  GPUBufferRecord gpu_buffer_entry = it->second;

  auto it2 = std::find(gpu_buffer_entry.access_ids.begin(), gpu_buffer_entry.access_ids.end(), req->access_id());
  if (it2 == gpu_buffer_entry.access_ids.end()) {
    spdlog::warn("Access ID not found");

    auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponseDirect(builder, false, "Acceess ID not found");
    auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp.o);
    builder.Finish(msg);
    return;
  }

  // remove access id
  gpu_buffer_entry.access_ids.erase(it2);

  // update last activity timestamp
  gpu_buffer_entry.last_activity_timestamp = std::chrono::steady_clock::now();

  // decrement access counter
  gpu_buffer_entry.access_counter--;

  // Delete the GPU buffer if it has no remaining accessors and expiration is ACCESS-based
  // if (gpu_buffer_entry.expiration_type == fbs::cuda::ipc::api::ExpirationOptionsType_ACCESS &&
  //     gpu_buffer_entry.access_counter == 0 &&
  //     gpu_buffer_entry.access_ids.empty()) {
  //   CudaUtils::FreeDeviceBuffer(gpu_buffer_entry.d_ptr);
  //   buffers_.erase(buffer_id);
  // }
}

boost::uuids::uuid CudaIPCServer::generateUUID() {
  // static ensures the generator is constructed only once
  static boost::uuids::random_generator gen;
  return gen();
}

void CudaIPCServer::expirationLoop() {
  using namespace std::chrono_literals;
  while (running_) {
    cleanupExpiredBuffers();
    std::this_thread::sleep_for(5s); // Run every second
  }
}

void CudaIPCServer::cleanupExpiredBuffers() {
  spdlog::trace("cleanupExpiredBuffers");
  auto                        now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(buffers_mutex_);

  for (auto it = buffers_.begin(); it != buffers_.end();) {
    auto& record = it->second;

    // Example expiration rule:
    if (record.expiration_type == fbs::cuda::ipc::api::ExpirationOptionsType_TIMESTAMP) {
      if (now >= record.expiration_timestamp) {
        spdlog::info("Buffer expired. Releasing GPU memory.");

        // Free CUDA memory
        CudaUtils::FreeDeviceBuffer(record.d_ptr);

        it = buffers_.erase(it);
      }
    } else if (record.expiration_type == fbs::cuda::ipc::api::ExpirationOptionsType_ACCESS) {
      if (record.access_counter == 0) {
        spdlog::info("Buffer expired due to zero access count.");

        // Free CUDA memory
        CudaUtils::FreeDeviceBuffer(record.d_ptr);
        it = buffers_.erase(it);
      }
    }

    ++it;
  }
}