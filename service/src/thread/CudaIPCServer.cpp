#include "CudaIPCServer.h"

CudaIPCServer::CudaIPCServer(const fbs::cuda::ipc::service::Configuration* configuration)
    : configuration_(configuration), context_(1), socket_(context_, zmq::socket_type::rep), running_(false) {
  // Init GPU Device
  // CudaUtils::InitDevice(configuration->gpu_device_index());

  // create an http server for Prometheus metrics
  exposer_ = std::make_unique<prometheus::Exposer>(configuration->prometheus_endpoint()->str());

  // create a metrics registry
  registry_ = std::make_shared<prometheus::Registry>();

  // Define metrics
  requests_total_ =
      &prometheus::BuildCounter().Name("cuda_ipc_requests_total").Help("Total number of IPC requests received").Register(*registry_).Add({});

  create_buffer_success_ = &prometheus::BuildCounter()
                                .Name("cuda_ipc_create_buffer_success_total")
                                .Help("Number of successfully created GPU buffers")
                                .Register(*registry_)
                                .Add({});

  create_buffer_fail_ = &prometheus::BuildCounter()
                             .Name("cuda_ipc_create_buffer_fail_total")
                             .Help("Number of failed GPU buffer creation attempts")
                             .Register(*registry_)
                             .Add({});

  allocated_buffers_ =
      &prometheus::BuildGauge().Name("cuda_ipc_allocated_buffers").Help("Current number of allocated GPU buffers").Register(*registry_).Add({});

  allocated_bytes_ =
      &prometheus::BuildGauge().Name("cuda_ipc_allocated_bytes").Help("Total GPU memory allocated in bytes").Register(*registry_).Add({});

  expired_buffers_ =
      &prometheus::BuildCounter().Name("cuda_ipc_expired_buffers_total").Help("Number of GPU buffers that have expired").Register(*registry_).Add({});

  // Histogram buckets (in seconds)
  std::vector<double> latency_buckets{
      1e-9,   // 1 ns
      10e-9,  // 10 ns
      100e-9, // 100 ns
      1e-6,   // 1 μs
      10e-6,  // 10 μs
      100e-6, // 100 μs
      1e-3,   // 1 ms
      10e-3,  // 10 ms
      0.1,    // 100 ms
      1.0     // 1 s
  };

  create_buffer_latency_ = &prometheus::BuildHistogram()
                                .Name("cuda_ipc_create_buffer_latency_seconds")
                                .Help("Latency of CreateBuffer requests in seconds")
                                .Register(*registry_)
                                .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  get_buffer_latency_ = &prometheus::BuildHistogram()
                             .Name("cuda_ipc_get_buffer_latency_seconds")
                             .Help("Latency of GetBuffer requests in seconds")
                             .Register(*registry_)
                             .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  notify_done_latency_ = &prometheus::BuildHistogram()
                              .Name("cuda_ipc_notify_done_latency_seconds")
                              .Help("Latency of NotifyDone requests in seconds")
                              .Register(*registry_)
                              .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  // Register the registry with exposer
  exposer_->RegisterCollectable(registry_);

  // bind zmq req socket
  socket_.bind(configuration->zmq_request_endpoint()->str());

  // set socket timeout so we dont block forever
  socket_.set(zmq::sockopt::rcvtimeo, 500);

  spdlog::info("Server listening on {}", configuration->zmq_request_endpoint()->str());
}

CudaIPCServer::~CudaIPCServer() {
  if (server_thread_.joinable()) server_thread_.join();
 // if (expiration_thread_.joinable()) expiration_thread_.join();
}

void CudaIPCServer::start() {
  running_ = true;

  // Start server zmq server thread
  server_thread_ = std::thread(&CudaIPCServer::run, this);

  // Start expiration cleanup thread
  //expiration_thread_ = std::thread(&CudaIPCServer::expirationLoop, this);
}

void CudaIPCServer::stop() {
  running_ = false;
}

void CudaIPCServer::join() {
  if (server_thread_.joinable()) server_thread_.join();
  //if (expiration_thread_.joinable()) expiration_thread_.join();
}

void CudaIPCServer::run() {
  while (running_) {
    zmq::message_t request_msg;
    spdlog::trace("Waiting for request...");
    auto recv_result = socket_.recv(request_msg, zmq::recv_flags::none);
    if (!recv_result) continue;

    // we have a request message
    spdlog::trace("Received message: {}", request_msg.size());

    // Increment total requests
    requests_total_->Increment();

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
    spdlog::trace("Sending response : {}", builder.GetSize());
    socket_.send(zmq::buffer(builder.GetBufferPointer(), builder.GetSize()), zmq::send_flags::none);
  }
}

void CudaIPCServer::handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  auto                        start = std::chrono::steady_clock::now();
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
  resp.add_access_id(access_id);

  try {
    // allocate device buffer and get handle
    auto d_ptr                  = CudaUtils::AllocDeviceBuffer(req->size());
    auto cuda_ipc_memory_handle = CudaUtils::GetCudaMemoryHandle(d_ptr);

    // create entry and save device pointer, size and handle
    GPUBufferRecord entry;
    entry.uuid       = buffer_id;
    entry.d_ptr      = d_ptr;
    entry.size       = req->size();
    entry.ipc_handle = cuda_ipc_memory_handle;
    entry.access_ids.push_back(access_id);
    entry.creation_timestamp      = std::chrono::steady_clock::now();
    entry.last_activity_timestamp = std::chrono::steady_clock::now();

    if (req->expiration()) {
      if (req->expiration()->option_type() == fbs::cuda::ipc::api::ExpirationOptions_TtlCreationOption) {
        entry.expiration_timestamp = std::chrono::steady_clock::now() + std::chrono::seconds(req->expiration()->option_as_TtlCreationOption()->ttl());
      }

      if (req->expiration()->option_type() == fbs::cuda::ipc::api::ExpirationOptions_AccessCountOption) {
        entry.access_counter = req->expiration()->option_as_AccessCountOption()->aceess_count();
      }
    }

    // add entry to buffers_ hash map
    buffers_[buffer_id] = entry;

    // save ipc handle to response
    resp.add_ipc_handle(&cuda_ipc_memory_handle);
    resp.add_success(true);

    // metrics update
    create_buffer_success_->Increment();
    allocated_buffers_->Set(buffers_.size());
    allocated_bytes_->Increment(req->size());
  } catch (const std::exception& e) {
    resp.add_success(false);
    // resp.add_error()
    create_buffer_fail_->Increment();
  }

  // finish up response message
  auto resp_offset = resp.Finish();
  auto msg         = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_CreateCUDABufferResponse, resp_offset.o);
  builder.Finish(msg);

  // metrics update
  auto                          end     = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  create_buffer_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  auto                        start = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(buffers_mutex_);

  // convert flatbuffers uuid to boost uuid
  auto buffer_id = util::UUIDConverter::toBoostUUID(*req->buffer_id());

  // search for buffer by buffer_id uuid
  auto it = buffers_.find(buffer_id);
  if (it == buffers_.end()) {
    spdlog::warn("Buffer not found");

    // return error
    auto resp = fbs::cuda::ipc::api::CreateGetCUDABufferResponseDirect(builder, nullptr, it->second.size, false, "Buffer not found");
    auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse, resp.o);
    builder.Finish(msg);

    auto end = std::chrono::steady_clock::now();
    get_buffer_latency_->Observe(std::chrono::duration<double>(end - start).count());
    return;
  }

  // get buffer entry in hashmap
  GPUBufferRecord gpu_buffer_entry = it->second;
  auto            access_id        = rand();
  gpu_buffer_entry.access_ids.push_back(access_id);                            // save access id
  gpu_buffer_entry.last_activity_timestamp = std::chrono::steady_clock::now(); // update last activity timestamp

  // init flatbuffers success response
  auto resp = fbs::cuda::ipc::api::CreateGetCUDABufferResponse(builder, &gpu_buffer_entry.ipc_handle, it->second.size, true);
  auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse, resp.o);
  builder.Finish(msg);

  // metrics update
  auto                          end     = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  get_buffer_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest* req, flatbuffers::FlatBufferBuilder& builder) {
  auto                        start = std::chrono::steady_clock::now();
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

    auto end = std::chrono::steady_clock::now();
    notify_done_latency_->Observe(std::chrono::duration<double>(end - start).count());
    return;
  }

  // get buffer entry in hashmap
  GPUBufferRecord gpu_buffer_entry = it->second;

  auto it2 = std::find(gpu_buffer_entry.access_ids.begin(), gpu_buffer_entry.access_ids.end(), req->access_id());
  if (it2 == gpu_buffer_entry.access_ids.end()) {
    spdlog::warn("Access ID not found");

    auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponseDirect(builder, false, "Access ID not found");
    auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp.o);
    builder.Finish(msg);

    auto end = std::chrono::steady_clock::now();
    notify_done_latency_->Observe(std::chrono::duration<double>(end - start).count());
    return;
  }

  // remove access id
  gpu_buffer_entry.access_ids.erase(it2);

  // update last activity timestamp
  gpu_buffer_entry.last_activity_timestamp = std::chrono::steady_clock::now();

  // decrement access counter
  gpu_buffer_entry.access_counter--;

  // init flatbuffers success response
  auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponseDirect(builder, true);
  auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp.o);
  builder.Finish(msg);

  // metrics update
  auto                          end     = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  notify_done_latency_->Observe(elapsed.count());
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

    // check for expiration
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

        // update metrics
        allocated_bytes_->Decrement(record.size);
        expired_buffers_->Increment();
        allocated_buffers_->Set(buffers_.size());

        // remove from hash map
        it = buffers_.erase(it);
      }
    }

    ++it;
  }
}