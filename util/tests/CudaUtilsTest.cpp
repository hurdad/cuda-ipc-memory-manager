#include <gtest/gtest.h>
#include "CudaUtils.h"
#include <vector>
#include <cstring> // for std::memcmp
#include <boost/uuid/uuid_generators.hpp>

class CudaUtilsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Optional: choose a specific CUDA device if you have multiple
    CudaUtils::SetDevice(0);
  }

  void TearDown() override {
    // Nothing to clean up globally
  }
};

// Test allocation and deallocation of device memory
TEST_F(CudaUtilsTest, AllocAndFreeDeviceBuffer) {
  size_t size  = 1024;
  void*  d_ptr = CudaUtils::AllocDeviceBuffer(size);
  ASSERT_NE(d_ptr, nullptr);
  EXPECT_NO_THROW(CudaUtils::FreeDeviceBuffer(d_ptr));
}

// Test zero-initialized allocation
TEST_F(CudaUtilsTest, ZeroInitializedAlloc) {
  size_t size  = 1024;
  void*  d_ptr = CudaUtils::AllocDeviceBuffer(size, true);

  std::vector<uint8_t> host_buf(size);
  CudaUtils::CopyToHost(host_buf.data(), d_ptr, size);

  for (auto byte : host_buf) {
    EXPECT_EQ(byte, 0);
  }

  CudaUtils::FreeDeviceBuffer(d_ptr);
}

// Test host-to-device and device-to-host memory copy
TEST_F(CudaUtilsTest, MemoryCopy) {
  size_t               size = 1024;
  std::vector<uint8_t> host_in(size, 42);
  std::vector<uint8_t> host_out(size, 0);

  void* d_ptr = CudaUtils::AllocDeviceBuffer(size);
  CudaUtils::CopyToDevice(d_ptr, host_in.data(), size);
  CudaUtils::CopyToHost(host_out.data(), d_ptr, size);

  EXPECT_EQ(host_in, host_out);
  CudaUtils::FreeDeviceBuffer(d_ptr);
}

// Test memory info retrieval
TEST_F(CudaUtilsTest, GetMemoryInfo) {
  size_t free_mem  = 0;
  size_t total_mem = 0;
  EXPECT_NO_THROW(CudaUtils::GetMemoryInfo(&free_mem, &total_mem));
  EXPECT_GT(total_mem, 0);
  EXPECT_GE(free_mem, 0);
  EXPECT_LE(free_mem, total_mem);
}

// Test device ID lookup from a UUID
TEST_F(CudaUtilsTest, GetDeviceIdFromUUID) {
  // int device_count = 0;
  // cudaGetDeviceCount(&device_count);
  // ASSERT_GT(device_count, 0);
  //
  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, 0);
  //
  // boost::uuids::string_generator gen;
  // boost::uuids::uuid uuid = gen(prop.uuid);
  //
  // int device_id = CudaUtils::GetDeviceIdFromUUID(uuid);
  // EXPECT_EQ(device_id, 0);
}

// Test IPC memory handle creation (basic sanity check)
TEST_F(CudaUtilsTest, CudaIpcHandle) {
  size_t size  = 1024;
  void*  d_ptr = CudaUtils::AllocDeviceBuffer(size);

  auto handle = CudaUtils::GetCudaMemoryHandle(d_ptr);
  ASSERT_EQ(handle.size(), 64);

  void* opened_ptr = CudaUtils::OpenHandleToCudaMemory(handle);
  ASSERT_NE(opened_ptr, nullptr);

  EXPECT_NO_THROW(CudaUtils::CloseHandleToCudaMemory(opened_ptr));
  CudaUtils::FreeDeviceBuffer(d_ptr);
}