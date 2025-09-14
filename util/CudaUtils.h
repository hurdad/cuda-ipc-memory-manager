#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "api/common_generated.h"

class CudaUtils {
public:
  /**
   * @brief Initalize CUDA device.
   * @param device_id
   */
  static void InitDevice(const int device_id) ;

  /**
   * @brief Allocates a buffer on the CUDA device.
   * @param numBytes Number of bytes to allocate.
   * @return Pointer to the device buffer, or nullptr on failure.
   */
  static void* AllocDeviceBuffer(size_t numBytes);

  /**
   * @brief Frees a previously allocated device buffer.
   * @param d_buffer Pointer to the device buffer.
   */
  static void FreeDeviceBuffer(void* d_buffer);

  /**
   * @brief Copies data from host to device.
   * @param d_buffer Device buffer pointer.
   * @param h_buffer Host buffer pointer.
   * @param numBytes Number of bytes to copy.
   * @return True if successful, false otherwise.
   */
  static bool CopyToDevice(void* d_buffer, const void* h_buffer, size_t numBytes);

  /**
   * @brief Copies data from device to host.
   * @param h_buffer Host buffer pointer.
   * @param d_buffer Device buffer pointer.
   * @param numBytes Number of bytes to copy.
   * @return True if successful, false otherwise.
   */
  static bool CopyToHost(void* h_buffer, const void* d_buffer, size_t numBytes);

  /**
   * @brief Creates a FlatBuffers CUDA IPC handle from a device pointer.
   * @param d_ptr Device pointer.
   * @return FlatBuffers CudaIPCHandle.
   * @throws std::runtime_error if CUDA IPC handle creation fails.
   */
  static fbs::cuda::ipc::api::CudaIPCHandle GetCudaMemoryHandle(void* d_ptr);

  /**
   * @brief Opens a CUDA IPC handle and returns the device pointer.
   * @param cuda_ipc_handle FlatBuffers CudaIPCHandle.
   * @return Device pointer corresponding to the IPC handle.
   * @throws std::runtime_error if opening the IPC handle fails.
   */
  static void* HandleToCudaMemory(const fbs::cuda::ipc::api::CudaIPCHandle& cuda_ipc_handle);
};

#endif // CUDA_UTILS_H