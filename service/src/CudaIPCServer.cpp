#include "CudaIPCServer.h"

CudaIPCServer::CudaIPCServer(const fbs::cuda::ipc::service::Configuration* configuration)
  : configuration_(configuration), context_(1), socket_(context_, zmq::socket_type::rep), running_(false) {
  // create a metrics registry
  registry_ = std::make_shared<prometheus::Registry>();

  // loop configured cuda gpu devices
  auto cuda_gpu_devices = configuration->cuda_gpu_devices();
  if (cuda_gpu_devices) {
    for (auto cuda_gpu_device : *cuda_gpu_devices) {
      // convert gpu_uuid to boost uuid
      boost::uuids::string_generator gen;
      auto                           boost_gpu_uuid = gen(cuda_gpu_device->gpu_uuid()->str());

      // lookup cuda_device_id from gpu_uuid with CUDA
      int cuda_device_id = CudaUtils::GetDeviceIdFromUUID(boost_gpu_uuid);
      spdlog::info("Found CUDA GPU = {} with cuda_device_id = {}", cuda_gpu_device->gpu_uuid()->str(), cuda_device_id);

      // save lookup
      gpu_uuid_to_cuda_device_index_[boost_gpu_uuid] = cuda_device_id;

      // CUDA memory info for this process
      size_t freeMem = 0, totalMem = 0;
      CudaUtils::SetDevice(cuda_device_id);
      CudaUtils::GetMemoryInfo(&freeMem, &totalMem);
      spdlog::info("CUDA GPU Device ID = {} Process GPU Memory Free = {:.2f} GB, Total = {:.2f} GB", cuda_device_id,
                   static_cast<double>(freeMem) / (1024.0 * 1024.0 * 1024.0), static_cast<double>(totalMem) / (1024.0 * 1024.0 * 1024.0));

      // Check max memory allocation type and set max_gpu_allocated_memory_
      auto max_allocation_type = cuda_gpu_device->max_memory_allocation_type();
      if (max_allocation_type) {
        switch (max_allocation_type) {
          case fbs::cuda::ipc::service::MemoryMaxAllocation_MaxFixedMemoryBytes:
            max_gpu_allocated_memory_[boost_gpu_uuid] = cuda_gpu_device->max_memory_allocation_as_MaxFixedMemoryBytes()->value();
            break;

          case fbs::cuda::ipc::service::MemoryMaxAllocation_MaxGPUMemoryPercentage:
            max_gpu_allocated_memory_[boost_gpu_uuid] = totalMem * cuda_gpu_device->max_memory_allocation_as_MaxGPUMemoryPercentage()->value();
            break;

          default:
            throw std::runtime_error("Unknown Max Memory Allocation Type");
        }
      }

      // init metrics per gpu
      allocated_buffers_map_[boost_gpu_uuid] = &prometheus::BuildGauge()
                                                .Name("cuda_ipc_allocated_buffers")
                                                .Help("Current number of buffers allocated by GPU")
                                                .Register(*registry_)
                                                .Add({{"uuid", cuda_gpu_device->gpu_uuid()->str()}});
      allocated_bytes_map_[boost_gpu_uuid] = &prometheus::BuildGauge()
                                              .Name("cuda_ipc_allocated_bytes")
                                              .Help("Current number of bytes allocated by GPU")
                                              .Register(*registry_)
                                              .Add({{"uuid", cuda_gpu_device->gpu_uuid()->str()}});
    }
  }

  // create an http server for Prometheus metrics
  exposer_ = std::make_unique<prometheus::Exposer>(configuration->prometheus_endpoint()->str());

  // Define metrics
  requests_total_ =
      &prometheus::BuildCounter().Name("cuda_ipc_api_requests").Help("Total number of IPC requests received").Register(*registry_).Add({});

  errors_total_ =
      &prometheus::BuildCounter().Name("cuda_ipc_api_errors").Help("Total number of IPC errors that have occured").Register(*registry_).Add({});

  expired_buffers_ =
      &prometheus::BuildCounter().Name("cuda_ipc_expired_buffers").Help("Number of GPU buffers that have expired").Register(*registry_).Add({});

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
                            .Name("cuda_ipc_api_create_buffer_latency_seconds")
                            .Help("Latency of CreateCUDABuffer requests in seconds")
                            .Register(*registry_)
                            .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  get_buffer_latency_ = &prometheus::BuildHistogram()
                         .Name("cuda_ipc_api_get_buffer_latency_seconds")
                         .Help("Latency of GetCUDABuffer requests in seconds")
                         .Register(*registry_)
                         .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  notify_done_latency_ = &prometheus::BuildHistogram()
                          .Name("cuda_ipc_api_notify_done_latency_seconds")
                          .Help("Latency of NotifyDone requests in seconds")
                          .Register(*registry_)
                          .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  free_buffer_latency_ = &prometheus::BuildHistogram()
                          .Name("cuda_ipc_api_free_buffer_latency_seconds")
                          .Help("Latency of FreeCUDABuffer requests in seconds")
                          .Register(*registry_)
                          .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  expire_buffers_latency_ = &prometheus::BuildHistogram()
                             .Name("cuda_ipc_expire_buffers_latency_seconds")
                             .Help("Latency of cleanupExpiredBuffers function requests in seconds")
                             .Register(*registry_)
                             .Add({}, prometheus::Histogram::BucketBoundaries{latency_buckets});

  // Register the registry with exposer
  exposer_->RegisterCollectable(registry_);

  // Register GPU Metrics Collector directly with the exposer
  gpu_metrics_collector_ = std::make_shared<GpuMetricsCollector>(configuration_);
  exposer_->RegisterCollectable(gpu_metrics_collector_);

  // Register Process Metrics Collector directly with the exposer
  process_metrics_collector_ = std::make_shared<ProcessMetricsCollector>();
  exposer_->RegisterCollectable(process_metrics_collector_);

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

      // verify flatbuffers IPC request
      flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), size);
      if (!fbs::cuda::ipc::api::VerifyRPCRequestMessageBuffer(verifier)) {
        throw std::runtime_error("Invalid IPC Request Message");
      }

      // parse flatbuffers request
      auto rpc_request = fbs::cuda::ipc::api::GetRPCRequestMessage(buf);
      auto req_type    = rpc_request->request_type();

      // call handle function for request type
      switch (req_type) {
        case fbs::cuda::ipc::api::RPCRequest_GetAvailableGPUsRequest:
          handleGetAvailableGPUs(rpc_request->request_as_GetAvailableGPUsRequest(), response_builder, start);
          break;

        case fbs::cuda::ipc::api::RPCRequest_GetAllocatedTotalBufferCountRequest:
          handleGetAllocatedTotalBufferCount(rpc_request->request_as_GetAllocatedTotalBufferCountRequest(), response_builder, start);
          break;

        case fbs::cuda::ipc::api::RPCRequest_GetAllocatedTotalBytesRequest:
          handleGetAllocatedTotalBytes(rpc_request->request_as_GetAllocatedTotalBytesRequest(), response_builder, start);
          break;

        case fbs::cuda::ipc::api::RPCRequest_GetMaxAllocationBytesRequest:
          handleGetMaxAllocationBytesRequest(rpc_request->request_as_GetMaxAllocationBytesRequest(), response_builder, start);
          break;

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

void CudaIPCServer::handleGetAvailableGPUs(const fbs::cuda::ipc::api::GetAvailableGPUsRequest* req,
                                           flatbuffers::FlatBufferBuilder&                     response_builder,
                                           std::chrono::time_point<std::chrono::steady_clock>  start_timestamp) {
  // list of flatbuffers gpu uuids to return
  std::vector<fbs::cuda::ipc::api::UUID> gpu_uuids;

  // loop configured cuda gpu devices
  auto cuda_gpu_devices = configuration_->cuda_gpu_devices();
  if (cuda_gpu_devices) {
    for (auto cuda_gpu_device : *cuda_gpu_devices) {
      // convert gpu_uuid to boost uuid
      boost::uuids::string_generator gen;
      auto                           boost_gpu_uuid = gen(cuda_gpu_device->gpu_uuid()->str());

      // add uuid to list
      gpu_uuids.push_back(util::UUIDConverter::toFlatBufferUUID(boost_gpu_uuid));
    }
  }

  // init flatbuffers response
  auto resp_offset = fbs::cuda::ipc::api::CreateGetAvailableGPUsResponseDirect(response_builder, &gpu_uuids);

  // finish up response message
  auto msg =
      fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_GetAvailableGPUsResponse, resp_offset.o);
  response_builder.Finish(msg);
}

void CudaIPCServer::handleGetAllocatedTotalBufferCount(const fbs::cuda::ipc::api::GetAllocatedTotalBufferCountRequest* req,
                                                       flatbuffers::FlatBufferBuilder&                                 response_builder,
                                                       std::chrono::time_point<std::chrono::steady_clock>              start_timestamp) {
  // convert request gpu uuid to boost uuid
  auto gpu_uuid       = req->gpu_uuid();
  auto boost_gpu_uuid = util::UUIDConverter::toBoostUUID(*gpu_uuid);

  // lookup allocated_buffers metric
  auto allocated_buffers_it = allocated_buffers_map_.find(boost_gpu_uuid);
  if (allocated_buffers_it == allocated_buffers_map_.end()) {
    throw std::runtime_error("GPU UUID not found!");
  }
  auto value = allocated_buffers_it->second->Value();

  // init flatbuffers response
  auto resp_offset = fbs::cuda::ipc::api::CreateGetAllocatedTotalBufferCountResponse(response_builder, static_cast<uint64_t>(value));

  // finish up response message
  auto msg =
      fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_GetAllocatedTotalBufferCountResponse,
                                                    resp_offset.o);
  response_builder.Finish(msg);
}

void CudaIPCServer::handleGetAllocatedTotalBytes(const fbs::cuda::ipc::api::GetAllocatedTotalBytesRequest* req,
                                                 flatbuffers::FlatBufferBuilder&                           response_builder,
                                                 std::chrono::time_point<std::chrono::steady_clock>        start_timestamp) {
  // convert request gpu uuid to boost uuid
  auto gpu_uuid       = req->gpu_uuid();
  auto boost_gpu_uuid = util::UUIDConverter::toBoostUUID(*gpu_uuid);

  // lookup allocated_bytes metric
  auto allocated_bytes_it = allocated_bytes_map_.find(boost_gpu_uuid);
  if (allocated_bytes_it == allocated_bytes_map_.end()) {
    throw std::runtime_error("GPU UUID not found!");
  }
  auto value = allocated_bytes_it->second->Value();

  // init flatbuffers response
  auto resp_offset = fbs::cuda::ipc::api::CreateGetAllocatedTotalBytesResponse(response_builder, static_cast<uint64_t>(value));

  // finish up response message
  auto msg =
      fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_GetAllocatedTotalBytesResponse, resp_offset.o);
  response_builder.Finish(msg);
}

void CudaIPCServer::handleGetMaxAllocationBytesRequest(const fbs::cuda::ipc::api::GetMaxAllocationBytesRequest* req,
                                                       flatbuffers::FlatBufferBuilder&                          response_builder,
                                                       std::chrono::time_point<std::chrono::steady_clock>       start_timestamp) {
  // convert request gpu uuid to boost uuid
  auto gpu_uuid       = req->gpu_uuid();
  auto boost_gpu_uuid = util::UUIDConverter::toBoostUUID(*gpu_uuid);

  // lookup max_gpu_allocated_memory_
  auto max_gpu_allocated_memory_it = max_gpu_allocated_memory_.find(boost_gpu_uuid);
  if (max_gpu_allocated_memory_it == max_gpu_allocated_memory_.end()) {
    throw std::runtime_error("GPU UUID not found!");
  }
  auto value = max_gpu_allocated_memory_it->second;

  // init flatbuffers response
  auto resp_offset = fbs::cuda::ipc::api::CreateGetMaxAllocationBytesResponse(response_builder, value);

  // finish up response message
  auto msg =
      fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_GetMaxAllocationBytesResponse, resp_offset.o);
  response_builder.Finish(msg);
}

void CudaIPCServer::handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& response_builder,
                                       std::chrono::time_point<std::chrono::steady_clock>  start_timestamp) {
  // convert request gpu uuid to boost uuid
  auto gpu_uuid       = req->gpu_uuid();
  auto boost_gpu_uuid = util::UUIDConverter::toBoostUUID(*gpu_uuid);
  spdlog::debug("Create Buffer on GPU UUID : " + boost::uuids::to_string(boost_gpu_uuid));

  // find max allocated memory for requested gpu_uuid
  auto max_gpu_allocated_it = max_gpu_allocated_memory_.find(boost_gpu_uuid);
  if (max_gpu_allocated_it == max_gpu_allocated_memory_.end()) {
    throw std::runtime_error("Unable to find GPU UUID : " + boost::uuids::to_string(boost_gpu_uuid));
  }

  // check if we have enough memory left (allocated + requested > total mem)
  if (allocated_bytes_map_[boost_gpu_uuid]->Value() + req->size() > max_gpu_allocated_it->second) {
    throw std::runtime_error("Out of Memory: Not enough free memory to satisfy the request.");
  }

  // set device before allocate
  int device_id = gpu_uuid_to_cuda_device_index_[boost_gpu_uuid];
  CudaUtils::SetDevice(device_id);

  // allocate device buffer and get handle
  auto d_ptr                      = CudaUtils::AllocDeviceBuffer(req->size(), req->zero_buffer());
  auto cuda_ipc_memory_handle_arr = CudaUtils::GetCudaMemoryHandle(d_ptr);

  // Convert CUDA handle to Flatbuffers span
  flatbuffers::span<const uint8_t, 64> fb_span(reinterpret_cast<const uint8_t*>(cuda_ipc_memory_handle_arr.data()), 64);
  fbs::cuda::ipc::api::CudaIPCHandle   cuda_ipc_memory_handle(fb_span);

  // generate ids
  auto buffer_id = generateUUID();
  auto access_id = rand();

  // convert to uuid flatbuffer
  auto uuid_flatbuffer = util::UUIDConverter::toFlatBufferUUID(buffer_id);

  // init flatbuffers response
  auto resp = fbs::cuda::ipc::api::CreateCUDABufferResponseBuilder(response_builder);
  resp.add_buffer_id(&uuid_flatbuffer);
  resp.add_size(req->size());
  resp.add_access_id(access_id);
  resp.add_ipc_handle(&cuda_ipc_memory_handle);
  resp.add_cuda_device_id(device_id);

  // create entry and save device pointer, size and handle
  GPUBufferRecord new_entry;
  new_entry.buffer_id             = buffer_id;
  new_entry.gpu_uuid              = boost_gpu_uuid;
  new_entry.d_ptr                 = d_ptr;
  new_entry.size                  = req->size();
  new_entry.ipc_handle            = cuda_ipc_memory_handle;
  new_entry.cuda_gpu_device_index = device_id;
  new_entry.access_ids.push_back(access_id);
  new_entry.creation_timestamp      = std::chrono::steady_clock::now();
  new_entry.last_activity_timestamp = std::chrono::steady_clock::now();

  // config expiration options
  if (req->expiration()) {
    // set expiration timestamp in the future (now + ttl)
    if (req->expiration()->ttl() && (req->expiration()->ttl() > 0)) {
      new_entry.expiration_timestamp = std::chrono::steady_clock::now() + std::chrono::seconds(req->expiration()->ttl());
    }

    // set access count
    if (req->expiration()->access_count() && (req->expiration()->access_count() > 0)) {
      new_entry.access_count = req->expiration()->access_count();
    }

    // copy struct
    std::memcpy(&new_entry.expiration_option, req->expiration(), sizeof(new_entry.expiration_option));
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
  } // lock released here

  // metrics update
  auto allocated_buffers_it = allocated_buffers_map_.find(boost_gpu_uuid);
  if (allocated_buffers_it != allocated_buffers_map_.end()) {
    allocated_buffers_it->second->Increment();
  }
  auto allocated_bytes_it = allocated_bytes_map_.find(boost_gpu_uuid);
  if (allocated_bytes_it != allocated_bytes_map_.end()) {
    allocated_bytes_it->second->Increment(req->size());
  }

  // finish up response message
  auto resp_offset = resp.Finish();
  auto msg         =
      fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_CreateCUDABufferResponse, resp_offset.o);
  response_builder.Finish(msg);

  // metrics update
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;
  create_buffer_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest*   req, flatbuffers::FlatBufferBuilder& response_builder,
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
    auto resp = fbs::cuda::ipc::api::CreateGetCUDABufferResponse(response_builder, gpu_buffer_entry.cuda_gpu_device_index,
                                                                 &gpu_buffer_entry.ipc_handle, gpu_buffer_entry.size, access_id);
    auto msg = fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_GetCUDABufferResponse, resp.o);
    response_builder.Finish(msg);
  } // lock released here

  // metrics update
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;
  get_buffer_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest*      req, flatbuffers::FlatBufferBuilder& response_builder,
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

      // decrement access count if enabled
      if (record.expiration_option.access_count() > 0) record.access_count--;
    });
  } // lock released here

  // init flatbuffers success response
  auto resp = fbs::cuda::ipc::api::CreateNotifyDoneResponse(response_builder);
  auto msg  = fbs::cuda::ipc::api::CreateRPCResponseMessage(response_builder, fbs::cuda::ipc::api::RPCResponse_NotifyDoneResponse, resp.o);

  response_builder.Finish(msg);
  // metrics update
  std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_timestamp;
  notify_done_latency_->Observe(elapsed.count());
}

void CudaIPCServer::handleFreeBuffer(const fbs::cuda::ipc::api::FreeCUDABufferRequest*  req, flatbuffers::FlatBufferBuilder& response_builder,
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

    // check if anyone is still accessing buffer
    if (gpu_buffer_entry.access_ids.size() > 0) {
      throw std::runtime_error(fmt::format("Buffer ID still being accessed = {}", boost::uuids::to_string(buffer_id)));
    }

    // set device before free
    CudaUtils::SetDevice(gpu_buffer_entry.cuda_gpu_device_index);

    // delete buffer
    CudaUtils::FreeDeviceBuffer(gpu_buffer_entry.d_ptr);

    // update metrics
    auto allocated_buffers_it = allocated_buffers_map_.find(gpu_buffer_entry.gpu_uuid);
    if (allocated_buffers_it != allocated_buffers_map_.end()) {
      allocated_buffers_it->second->Decrement();
    }
    auto allocated_bytes_it = allocated_bytes_map_.find(gpu_buffer_entry.gpu_uuid);
    if (allocated_bytes_it != allocated_bytes_map_.end()) {
      allocated_bytes_it->second->Decrement(gpu_buffer_entry.size);
    }

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
  std::lock_guard<std::mutex> lock(buffers_mutex_);

  // --- TTL Expired Buffers ---
  auto& exp_index   = buffers_.get<ByExpiration>();
  auto  expired_end = exp_index.upper_bound(now);

  for (auto it = exp_index.begin(); it != expired_end; /* increment inside loop */) {
    GPUBufferRecord& buf = const_cast<GPUBufferRecord&>(*it);

    if (buf.expiration_option.ttl() > 0) {
      spdlog::info("Buffer expired by TTL. Releasing GPU memory: {}", boost::uuids::to_string(buf.buffer_id));

      // free buffer and erase
      CudaUtils::SetDevice(buf.cuda_gpu_device_index);
      CudaUtils::FreeDeviceBuffer(buf.d_ptr);
      buf.d_ptr = nullptr; // prevent dangling pointer

      // update metrics
      auto allocated_buffers_it = allocated_buffers_map_.find(buf.gpu_uuid);
      if (allocated_buffers_it != allocated_buffers_map_.end()) {
        allocated_buffers_it->second->Decrement();
      }
      auto allocated_bytes_it = allocated_bytes_map_.find(buf.gpu_uuid);
      if (allocated_bytes_it != allocated_bytes_map_.end()) {
        allocated_bytes_it->second->Decrement(buf.size);
      }
      expired_buffers_->Increment();

      it = exp_index.erase(it); // erase safely
    } else {
      ++it;
    }
  }

  // --- Access-count Expired Buffers ---
  auto& access_index = buffers_.get<ByAccessOptionAndCount>();

  uint64_t min_allowed  = 1;
  uint64_t current_zero = 0;

  auto it = access_index.lower_bound(std::make_pair(min_allowed, current_zero));

  while (it != access_index.end() && it->access_count == 0) {
    GPUBufferRecord& buf = const_cast<GPUBufferRecord&>(*it);
    spdlog::info("Buffer expired by access count. Releasing GPU memory: {}", boost::uuids::to_string(buf.buffer_id));

    // free buffer and erase
    CudaUtils::SetDevice(buf.cuda_gpu_device_index);
    CudaUtils::FreeDeviceBuffer(buf.d_ptr);
    buf.d_ptr = nullptr;

    // update metrics
    auto allocated_buffers_it = allocated_buffers_map_.find(buf.gpu_uuid);
    if (allocated_buffers_it != allocated_buffers_map_.end()) {
      allocated_buffers_it->second->Decrement();
    }
    auto allocated_bytes_it = allocated_bytes_map_.find(buf.gpu_uuid);
    if (allocated_bytes_it != allocated_bytes_map_.end()) {
      allocated_bytes_it->second->Decrement(buf.size);
    }
    expired_buffers_->Increment();

    it = access_index.erase(it); // erase safely
  }
}