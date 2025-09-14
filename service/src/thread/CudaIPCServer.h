#ifndef CUDA_IPC_SERVER_H
#define CUDA_IPC_SERVER_H

#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>
#include <string>
#include <zmq.hpp>
#include <spdlog/spdlog.h>

// Boost UUID
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/functional/hash.hpp>

// UUID Util
#include "UUIDConverter.hpp"

// CUDA Utils
#include "CudaUtils.h"

// FlatBuffers header generated from schema .fbs
#include "api/rpc_request_generated.h"
#include "api/rpc_response_generated.h"


struct GPUBufferRecord
{
    // GPU Buffer ID
    boost::uuids::uuid uuid;

    // CUDA IPC handle for sharing GPU memory across processes (flatbuffers)
    fbs::cuda::ipc::api::CudaIPCHandle ipc_handle;

    // How to handle expiration of this buffer
    fbs::cuda::ipc::api::ExpirationOptionsType expiration_type;

    // Device pointer to the GPU memory
    void* d_ptr = nullptr;

    // Size of the buffer in bytes
    size_t size = 0;

    // Number of access counts
    size_t access_counter = 0;

    // Creation Timestamp of the buffer
    std::chrono::steady_clock::time_point creation_timestamp;

    // Timestamp of the last activity on this buffer
    std::chrono::steady_clock::time_point last_activity_timestamp;

    // Timestamp of when to expire on this buffer
    std::chrono::steady_clock::time_point expiration_timestamp;

    // list of current access ids
    std::vector<uint32_t> access_ids;
};

class CudaIPCServer
{
public:
    CudaIPCServer(const std::string& endpoint);
    ~CudaIPCServer();

    void start();
    void stop();
    void join();

private:
    zmq::context_t context_;
    zmq::socket_t socket_;
    std::thread thread_;
    std::string endpoint_;
    std::atomic<bool> running_; // stop flag
    std::unordered_map<boost::uuids::uuid, GPUBufferRecord, boost::hash<boost::uuids::uuid>> buffers_;

    void run();

    // Functions for handling individual request types
    void handleCreateBuffer(const fbs::cuda::ipc::api::CreateCUDABufferRequest* req,
                            flatbuffers::FlatBufferBuilder& builder);
    void handleGetBuffer(const fbs::cuda::ipc::api::GetCUDABufferRequest* req, flatbuffers::FlatBufferBuilder& builder);
    void handleNotifyDone(const fbs::cuda::ipc::api::NotifyDoneRequest* req, flatbuffers::FlatBufferBuilder& builder);

    static boost::uuids::uuid generateUUID();
};

#endif // CUDA_UTILS_H
