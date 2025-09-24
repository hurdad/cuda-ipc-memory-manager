#include "CudaIPCServer.h"

CudaIPCServer::CudaIPCServer(const fbs::cuda::ipc::service::Configuration* configuration)
  : configuration_(configuration), context_(1), socket_(context_, zmq::socket_type::rep), running_(false) {
  // create an http server for Prometheus metrics
  exposer_ = std::make_unique<prometheus::Exposer>(configuration->prometheus_endpoint()->str());

  // create a metrics registry
  registry_ = std::make_shared<prometheus::Registry>();

  // Define metrics
  requests_total_ =
      &prometheus::BuildCounter().Name("cuda_ipc_requests_total").Help("Total number of IPC requests received").Register(*registry_).Add({});

  errors_total_ =
      &prometheus::BuildCounter().Name("cuda_ipc_errors_total").Help("Total number of IPC errors that have occured").Register(*registry_).Add({});

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
      1e-6, // 1 μs
      10e-6, // 10 μs
      100e-6, // 100 μs
      1e-3, // 1 ms
      10e-3, // 10 ms
      0.1, // 100 ms
      1.0 // 1 s
  };

  create_buffer_latency_ = &prometheus::BuildHistogram()
                            .Name("cuda_ipc_create_buffer_latency_seconds")
                            .Help("Latency of CreateCUDABuffer requests in seconds")
                            .Register(*registry_)
                            .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  get_buffer_latency_ = &prometheus::BuildHistogram()
                         .Name("cuda_ipc_get_buffer_latency_seconds")
                         .Help("Latency of GetCUDABuffer requests in seconds")
                         .Register(*registry_)
                         .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  notify_done_latency_ = &prometheus::BuildHistogram()
                          .Name("cuda_ipc_notify_done_latency_seconds")
                          .Help("Latency of NotifyDone requests in seconds")
                          .Register(*registry_)
                          .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  free_buffer_latency_ = &prometheus::BuildHistogram()
                          .Name("cuda_ipc_free_buffer_latency_seconds")
                          .Help("Latency of FreeCUDABuffer requests in seconds")
                          .Register(*registry_)
                          .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  expire_buffers_latency_= &prometheus::BuildHistogram()
                          .Name("cuda_ipc_expire_buffers_latency_seconds")
                          .Help("Latency of cleanupExpiredBuffers function requests in seconds")
                          .Register(*registry_)
                          .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  // Register the registry with exposer
  exposer_->RegisterCollectable(registry_);

  // Register GPU collector directly with the exposer
  exposer_->RegisterCollectable(std::make_shared<GpuMetricsCollector>());

  // bind zmq req socket
  socket_.bind(configuration->zmq_request_endpoint()->str());

  // set socket timeout so we dont block forever
  socket_.set(zmq::sockopt::rcvtimeo, 500);

  spdlog::info("Server listening on {}", configuration->zmq_request_endpoint()->str());
}

CudaIPCServer::~CudaIPCServer() {
  if (server_thread_.joinable()) server_thread_.join();
  if (expiration_thread_.joinable()) expiration_thread_.join();
}

void CudaIPCServer::start() {
  running_ = true;

  // Start server zmq server thread
  server_thread_ = std::thread(&CudaIPCServer::run, this);

  // Set Realtime Priority (if enabled)
  if (configuration_->set_realtime_priority()) {
    if (setThreadRealtime(server_thread_)) {
      spdlog::info("Successfully set server_thread to real-time priority!");
    } else {
      spdlog::warn("Failed to set server_thread real-time priority.");
    }
  }

  // Start expiration cleanup thread
  expiration_thread_ = std::thread(&CudaIPCServer::expirationLoop, this, configuration_->expiration_thread_interval_ms());

  // Set Realtime Priority (if enabled)
  if (configuration_->set_realtime_priority()) {
    if (setThreadRealtime(expiration_thread_)) {
      spdlog::info("Successfully set expiration_thread to real-time priority!");
    } else {
      spdlog::warn("Failed to set expiration_thread real-time priority.");
    }
  }
}

void CudaIPCServer::stop() {
  running_ = false;
}

void CudaIPCServer::join() {
  if (server_thread_.joinable()) server_thread_.join();
  if (expiration_thread_.joinable()) expiration_thread_.join();
}

void CudaIPCServer::run() {
  while (running_) {
    zmq::message_t request_msg;
    spdlog::trace("Waiting for request...");
    auto recv_result = socket_.recv(request_msg, zmq::recv_flags::none);
    if (!recv_result) continue;

    // start timestamp
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

    // we have a request message
    spdlog::trace("Received message: {}", request_msg.size());

    // builder for response
    flatbuffers::FlatBufferBuilder response_builder;

    try {
      // Increment total requests
      requests_total_->Increment();

      // get request buffer
      auto buf  = request_msg.data();
      auto size = request_msg.size();

      // verify flatbuffers request
      flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), size);
      if (!fbs::cuda::ipc::api::VerifyRPCRequestMessageBuffer(verifier)) {
        throw std::runtime_error("Invalid IPC Request Message");
      }

      // parse flatbuffers
      auto rpc_request = fbs::cuda::ipc::api::GetRPCRequestMessage(buf);
      auto req_type    = rpc_request->request_type();

      switch (req_type) {
        case fbs::cuda::ipc::api::RPCRequest_CreateCUDABufferRequest:
          handleCreateBuffer(rpc_request->request_as_CreateCUDABufferRequest(), response_builder, start);
          break;

        case fbs::cuda::ipc::api::RPCRequest_GetCUDABufferRequest:
          handleGetBuffer(rpc_request->request_as_GetCUDABufferRequest(), response_builder, start);
          break;

        case fbs::cuda::ipc::api::RPCRequest_NotifyDoneRequest:
          handleNotifyDone(rpc_request->request_as_NotifyDoneRequest(), response_builder, start);
          break;

        case fbs::cuda::ipc::api::RPCRequest_FreeCUDABufferRequest:
          handleFreeBuffer(rpc_request->request_as_FreeCUDABufferRequest(), response_builder, start);
          break;

        default:
          throw std::runtime_error("Unknown RPC Request Type");
      }
    } catch (const std::exception& e) {
      // log error
      spdlog::error(e.what());

      // build error response message
      auto resp = fbs::cuda::ipc::api::CreateErrorResponseDirect(response_builder, e.what());
      auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_ErrorResponse, resp.o);
      response_builder.Finish(msg);

      // increment errors
      errors_total_->Increment();
    }

    // send reply flatbuffers binary response
    spdlog::trace("Sending response : {}", response_builder.GetSize());
    auto result = socket_.send(zmq::buffer(response_builder.GetBufferPointer(), response_builder.GetSize()), zmq::send_flags::none);
    if (!result.has_value()) {
      spdlog::error("ZMQ Send failed. Error: {}", zmq_strerror(zmq_errno()));
    }
  }
}

void CudaIPCServer::handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req,
                                       flatbuffers::FlatBufferBuilder&                     response_builder,
                                       std::chrono::time_point<std::chrono::steady_clock>  start_timestamp) {
  // lookup gpu_device_index in config
  auto cuda_gpu_device = configuration_->cuda_gpu_devices()->LookupByKey(req->gpu_device_index());
  if (!cuda_gpu_device) {
    throw std::runtime_error(fmt::format("GPU Device ID not found = {}", req->gpu_device_index()));
  }

  // set device before allocate
  CudaUtils::SetDevice(req->gpu_device_index());

  // allocate device buffer and get handle
  auto d_ptr                  = CudaUtils::AllocDeviceBuffer(req->size(), req->zero_buffer());
  auto cuda_ipc_memory_handle = CudaUtils::GetCudaMemoryHandle(d_ptr);

  // init flatbuffers response
  auto resp = fbs::cuda::ipc::api::CreateCUDABufferResponseBuilder(response_builder);

  // generate ids
  auto buffer_id = generateUUID();
  auto access_id = rand();

  // convert to uuid flatbuffer
  auto uuid_flatbuffer = util::UUIDConverter::toFlatBufferUUID(buffer_id);

  // save to response
  resp.add_buffer_id(&uuid_flatbuffer);
  resp.add_access_id(access_id);
  resp.add_ipc_handle(&cuda_ipc_memory_handle);
  resp.add_gpu_device_index(req->gpu_device_index());

  // create entry and save device pointer, size and handle
  GPUBufferRecord new_entry;
  new_entry.buffer_id  = buffer_id;
  new_entry.d_ptr      = d_ptr;
  new_entry.size       = req->size();
  new_entry.ipc_handle = cuda_ipc_memory_handle;
  new_entry.access_ids.push_back(access_id);
  new_entry.creation_timestamp      = std::chrono::steady_clock::now();
  new_entry.last_activity_timestamp = std::chrono::steady_clock::now();

  if (req->expiration()) {
    new_entry.expiration_option = req->expiration()->option_type();
    if (req->expiration()->option_type() == fbs::cuda::ipc::api::ExpirationOptions_TtlCreationOption) {
      new_entry.expiration_timestamp = std::chrono::steady_clock::now() + std::chrono::seconds(
                                           req->expiration()->option_as_TtlCreationOption()->ttl());
      new_entry.expiration_option = fbs::cuda::ipc::api::ExpirationOptions_TtlCreationOption;
    }

    // if (req->expiration()->option_type() == fbs::cuda::ipc::api::ExpirationOptions_AccessCountOption) {
    //   entry.access_counter = req->expiration()->option_as_AccessCountOption()->aceess_count();
    // }
  }

  // lock buffers_mutex used to protect buffers_
  {
    std::lock_guard<std::mutex> lock(buffers_mutex_);

    // Insert new_entry into into the container
    auto result = buffers_.insert(std::move(new_entry));

    if (!result.second) {
      // insertion failed (duplicate buffer_id)
      throw std::runtime_error("Buffer with this ID already exists!");
    }

    // update metric while locked
    allocated_buffers_->Set(buffers_.size()); // update metric while locked
  } // lock released here

  // metrics update
  create_buffer_success_->Increment();
  allocated_bytes_->Increment(req->size());

  // finish up response message
  auto resp_offset = resp.Finish();
  auto msg         =
      fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_CreateCUDABufferResponse, resp_offset.o);
  response_builder.Finish(msg);

  // metrics update
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;
  create_buffer_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest*   req,
                                    flatbuffers::FlatBufferBuilder&                    response_builder,
                                    std::chrono::time_point<std::chrono::steady_clock> start_timestamp) {
  // convert flatbuffers uuid to boost uuid
  auto buffer_id = util::UUIDConverter::toBoostUUID(*req->buffer_id());

  // lock buffers_mutex used to protect buffers_
  {
    std::lock_guard<std::mutex> lock(buffers_mutex_);

    // search for buffer by buffer_id uuid using id index
    auto& id_index = buffers_.get<ByBufferId>();
    auto  it       = id_index.find(buffer_id);
    if (it == id_index.end()) {
      throw std::runtime_error(fmt::format("Buffer ID not found = {}", boost::uuids::to_string(buffer_id)));
    }

    // get buffer entry
    GPUBufferRecord& gpu_buffer_entry = const_cast<GPUBufferRecord&>(*it);

    // generate access id
    auto access_id = rand();

    // update buffer entry - add new access_id and update last activity timestamp
    id_index.modify(it, [access_id](GPUBufferRecord& record) {
      record.access_ids.push_back(access_id); // save access id
      record.last_activity_timestamp = std::chrono::steady_clock::now(); // update last activity timestamp
    });

    // init flatbuffers success response
    auto resp = fbs::cuda::ipc::api::CreateGetCUDABufferResponse(response_builder,
                                                                 gpu_buffer_entry.gpu_device_index,
                                                                 &gpu_buffer_entry.ipc_handle,
                                                                 gpu_buffer_entry.size,
                                                                 access_id);
    auto msg = fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder,
                                                             fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse,
                                                             resp.o);
    response_builder.Finish(msg);
  } // lock released here

  // metrics update
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;
  get_buffer_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest*      req,
                                     flatbuffers::FlatBufferBuilder&                    response_builder,
                                     std::chrono::time_point<std::chrono::steady_clock> start_timestamp) {
  // convert flatbuffers uuid to boost uuid
  auto buffer_id = util::UUIDConverter::toBoostUUID(*req->buffer_id());

  // lock buffers_mutex used to protect buffers_
  {
    std::lock_guard<std::mutex> lock(buffers_mutex_);

    // search for buffer by buffer_id uuid using id index
    auto& id_index = buffers_.get<ByBufferId>();
    auto  it       = id_index.find(buffer_id);
    if (it == id_index.end()) {
      throw std::runtime_error(fmt::format("Buffer ID not found = {}", boost::uuids::to_string(buffer_id)));
    }

    // get buffer entry
    GPUBufferRecord& gpu_buffer_entry = const_cast<GPUBufferRecord&>(*it);

    // search for access id
    auto access_id = std::find(gpu_buffer_entry.access_ids.begin(), gpu_buffer_entry.access_ids.end(), req->access_id());
    if (access_id == gpu_buffer_entry.access_ids.end()) {
      throw std::runtime_error(fmt::format("Access ID not found = {}", req->access_id()));
    }

    // update buffer entry - add new access_id and update last activity timestamp
    id_index.modify(it, [access_id](GPUBufferRecord& record) {
      record.access_ids.erase(access_id); // delete access_id from list
      record.last_activity_timestamp = std::chrono::steady_clock::now(); // update last activity timestamp
    });

    // decrement access counter
    // gpu_buffer_entry.access_counter--;
  } // lock released here

  // init flatbuffers success response
  auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponse(response_builder);
  auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp.o);

  response_builder.Finish(msg);
  // metrics update
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;
  notify_done_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleFreeBuffer(const fbs::cuda::ipc::api::FreeCUDABufferRequest*  req,
                                     flatbuffers::FlatBufferBuilder&                    response_builder,
                                     std::chrono::time_point<std::chrono::steady_clock> start_timestamp) {
  // convert flatbuffers uuid to boost uuid
  auto buffer_id = util::UUIDConverter::toBoostUUID(*req->buffer_id());

  // lock buffers_mutex used to protect buffers_
  {
    std::lock_guard<std::mutex> lock(buffers_mutex_);

    // search for buffer by buffer_id uuid using id index
    auto& id_index = buffers_.get<ByBufferId>();
    auto  it       = id_index.find(buffer_id);
    if (it == id_index.end()) {
      throw std::runtime_error(fmt::format("Buffer ID not found = {}", boost::uuids::to_string(buffer_id)));
    }

    // get buffer entry
    GPUBufferRecord& gpu_buffer_entry = const_cast<GPUBufferRecord&>(*it);

    // set device before free
    CudaUtils::SetDevice(gpu_buffer_entry.gpu_device_index);

    // delete buffer
    CudaUtils::FreeDeviceBuffer(gpu_buffer_entry.d_ptr);

    // update metrics
    allocated_bytes_->Decrement(gpu_buffer_entry.size);
    expired_buffers_->Increment();
    allocated_buffers_->Set(buffers_.size());

    // remove from buffers
    buffers_.erase(it);
  } // lock released here

  // init flatbuffers success response
  auto resp = fbs::cuda::ipc::api::CreateFreeCUDABufferResponse(response_builder);
  auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_FreeCUDABufferResponse, resp.o);
  response_builder.Finish(msg);

  // metrics update
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;
  free_buffer_latency_->Observe(elapsed.count());
}

boost::uuids::uuid CudaIPCServer::generateUUID() {
  // static ensures the generator is constructed only once
  static boost::uuids::random_generator gen;
  return gen();
}

void CudaIPCServer::expirationLoop(const uint32_t expiration_thread_interval_ms) {
  // Convert the interval from milliseconds to a chrono duration
  const auto interval = std::chrono::milliseconds(expiration_thread_interval_ms);

  // Main loop runs as long as the server is running
  while (running_) {
    // Record the start time of this iteration
    auto start_timestamp = std::chrono::steady_clock::now();

    // Perform cleanup of expired buffers
    cleanupExpiredBuffers();

    // Calculate how long the cleanupExpiredBuffers took
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;

    // metrics update
    expire_buffers_latency_->Observe(elapsed.count());

    // Sleep for the remaining time to maintain a consistent interval
    // If cleanup took longer than the interval, skip sleeping
    if (elapsed < interval) {
      std::this_thread::sleep_for(interval - elapsed);
    }
  }
}

bool CudaIPCServer::setThreadRealtime(std::thread& t) {
  // Get the native pthread handle from the std::thread
  pthread_t handle = t.native_handle();

  // Create a scheduling parameters structure
  sched_param sch_params;

  // Set the priority to the maximum allowed for SCHED_FIFO
  sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);

  // Apply real-time scheduling policy (SCHED_FIFO) with the specified priority
  // Returns true if successful, false otherwise
  return pthread_setschedparam(handle, SCHED_FIFO, &sch_params) == 0;
}

void CudaIPCServer::cleanupExpiredBuffers() {
  spdlog::trace("cleanupExpiredBuffers");

  auto                        now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(buffers_mutex_); // lock buffers_mutex used to protect buffers_

  // lookup expired payload by index
  auto& exp_index   = buffers_.get<ByExpiration>();
  auto  expired_end = exp_index.upper_bound(now);
  for (auto it = exp_index.begin(); it != expired_end; /* increment inside loop */) {
    // get buffer entry
    GPUBufferRecord& gpu_buffer_entry = const_cast<GPUBufferRecord&>(*it);

    // Only check buffers that use TTL expiration
    if (gpu_buffer_entry.expiration_option == fbs::cuda::ipc::api::ExpirationOptions_TtlCreationOption) {
      spdlog::info("Buffer expired. Releasing GPU memory. buffer_id = {}",
                   boost::uuids::to_string(gpu_buffer_entry.buffer_id));

      // Set CUDA device
      CudaUtils::SetDevice(gpu_buffer_entry.gpu_device_index);

      // Free CUDA memory
      CudaUtils::FreeDeviceBuffer(gpu_buffer_entry.d_ptr);

      // Update metrics before erasing
      allocated_bytes_->Decrement(gpu_buffer_entry.size);
      expired_buffers_->Increment();

      // Erase safely and get next iterator
      it = exp_index.erase(it);

      // Update allocated buffers metric
      allocated_buffers_->Set(buffers_.size());
    } else {
      ++it; // only increment if we didn't erase
    }
  }
}