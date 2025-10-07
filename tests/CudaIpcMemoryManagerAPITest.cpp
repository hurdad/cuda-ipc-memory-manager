#include <gtest/gtest.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "api/CudaIpcMemoryManagerAPI.h" // API HEADER

using namespace cuda::ipc::api;

class CudaIpcMemoryManagerAPITest : public ::testing::Test {
 protected:
  static inline std::string              test_endpoint = "ipc:///tmp/cuda-ipc-memory-manager-service.ipc";
  static inline CudaIpcMemoryManagerAPI* api           = nullptr;

  static void SetUpTestSuite() {
    // Construct the API object (assumes endpoint is valid)
    api = new CudaIpcMemoryManagerAPI(test_endpoint);
  }

  static void TearDownTestSuite() {
    delete api;
    api = nullptr;
  }
};

// ===============================
// GPU Query Tests
// ===============================
TEST_F(CudaIpcMemoryManagerAPITest, GetAvailableGPUsReturnsVector) {
  auto gpus = api->GetAvailableGPUs();
  EXPECT_FALSE(gpus.empty()) << "No GPUs detected on the system";
}

TEST_F(CudaIpcMemoryManagerAPITest, GetAllocatedTotalBufferCount) {
  auto     gpus   = api->GetAvailableGPUs();
  auto     gpu_id = gpus.front();
  uint64_t count  = api->GetAllocatedTotalBufferCount(gpu_id);
  EXPECT_GE(count, 0);
}

TEST_F(CudaIpcMemoryManagerAPITest, GetAllocatedTotalBytes) {
  auto     gpus   = api->GetAvailableGPUs();
  auto     gpu_id = gpus.front();
  uint64_t bytes  = api->GetAllocatedTotalBytes(gpu_id);
  EXPECT_GE(bytes, 0);
}

TEST_F(CudaIpcMemoryManagerAPITest, GetMaxAllocationBytes) {
  auto     gpus      = api->GetAvailableGPUs();
  auto     gpu_id    = gpus.front();
  uint64_t max_bytes = api->GetMaxAllocationBytes(gpu_id);
  EXPECT_GT(max_bytes, 0);
}

// ===============================
// GPUBuffer Management Tests
// ===============================
TEST_F(CudaIpcMemoryManagerAPITest, CreateAndGetCUDABufferRequest) {
  auto gpus   = api->GetAvailableGPUs();
  auto gpu_id = gpus.front();

  GPUBuffer buffer = api->CreateCUDABufferRequest(1024, gpu_id, 0, 10, true);

  EXPECT_NE(buffer.getDataPtr(), nullptr);
  EXPECT_EQ(buffer.getSize(), 1024);
  EXPECT_GT(buffer.getAccessId(), 0);
  EXPECT_NO_THROW(api->NotifyDoneRequest(buffer));

  // Fetch by buffer ID
  GPUBuffer fetched = api->GetCUDABufferRequest(buffer.getBufferId());
  EXPECT_EQ(fetched.getBufferId(), buffer.getBufferId());
  EXPECT_EQ(fetched.getSize(), buffer.getSize());
  EXPECT_EQ(fetched.getDataPtr(), buffer.getDataPtr());
  EXPECT_NO_THROW(api->NotifyDoneRequest(fetched));
}

TEST_F(CudaIpcMemoryManagerAPITest, NotifyDoneRequest) {
  auto gpus   = api->GetAvailableGPUs();
  auto gpu_id = gpus.front();

  GPUBuffer buffer = api->CreateCUDABufferRequest(512, gpu_id, 0, 10, false);
  EXPECT_NO_THROW(api->NotifyDoneRequest(buffer));
}

TEST_F(CudaIpcMemoryManagerAPITest, FreeCUDABufferRequest) {
  auto gpus   = api->GetAvailableGPUs();
  auto gpu_id = gpus.front();

  GPUBuffer buffer = api->CreateCUDABufferRequest(256, gpu_id, 0, 10, false);
  EXPECT_NO_THROW(api->NotifyDoneRequest(buffer));
  EXPECT_NO_THROW(api->FreeCUDABufferRequest(buffer.getBufferId()));

  // After freeing, attempting to fetch should throw (assumes API throws)
  EXPECT_THROW(api->GetCUDABufferRequest(buffer.getBufferId()), std::exception);
}
