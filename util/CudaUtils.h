#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <array>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_io.hpp>

/**
 * Macro to wrap CUDA API calls.
 * If the CUDA call fails, it throws a std::runtime_error with a descriptive message
 * including file and line number.
 *
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define CUDA_CHECK(call) do {                                 \
    cudaError_t status = (call);                              \
    if (status != cudaSuccess) {                               \
        throw std::runtime_error(                              \
            std::string("[CudaUtils] CUDA error at ") +        \
            __FILE__ + ":" + std::to_string(__LINE__) + " - " + \
            cudaGetErrorString(status));                       \
    }                                                          \
} while (0)

/**
 * Utility class for CUDA device management, memory allocation,
 * memory transfers, and CUDA IPC operations.
 */
class CudaUtils {
public:
  /**
   * Set the active CUDA device by ID.
   * @param device_id The CUDA device ID to select.
   * @throws std::runtime_error if cudaSetDevice fails.
   */
  static void SetDevice(int device_id);

  /**
    * Allocate a device buffer of the given size in bytes.
    * @param numBytes Number of bytes to allocate.
    * @param zeroInitialize If true, initializes the buffer to zero.
    * @return Pointer to the allocated device memory.
    * @throws std::runtime_error if cudaMalloc or cudaMemset fails.
    */
  static void* AllocDeviceBuffer(size_t numBytes, bool zeroInitialize = false);

  /**
   * Free a previously allocated device buffer.
   * @param d_buffer Pointer to device memory to free.
   * @throws std::runtime_error if cudaFree fails.
   */
  static void FreeDeviceBuffer(void* d_buffer);

  /**
   * Copy memory from host to device.
   * @param d_buffer Device buffer pointer.
   * @param h_buffer Host buffer pointer.
   * @param numBytes Number of bytes to copy.
   * @throws std::runtime_error if pointers are null or cudaMemcpy fails.
   */
  static void CopyToDevice(void* d_buffer, const void* h_buffer, size_t numBytes);

  /**
   * Copy memory from device to host.
   * @param h_buffer Host buffer pointer.
   * @param d_buffer Device buffer pointer.
   * @param numBytes Number of bytes to copy.
   * @throws std::runtime_error if pointers are null or cudaMemcpy fails.
   */
  static void CopyToHost(void* h_buffer, const void* d_buffer, size_t numBytes);

  /**
   * Create a CUDA IPC memory handle for sharing device memory between processes.
   * @param d_ptr Pointer to device memory.
   * @return Flatbuffers-wrapped IPC handle.
   * @throws std::runtime_error if cudaIpcGetMemHandle fails.
   */
  static const std::array<uint8_t, 64> GetCudaMemoryHandle(void* d_ptr);

  /**
   * Open a CUDA IPC handle in the current process.
   * @param cuda_ipc_handle IPC handle from another process.
   * @return Pointer to device memory mapped in this process.
   * @throws std::runtime_error if cudaIpcOpenMemHandle fails.
   */
  static void* OpenHandleToCudaMemory(const std::array<uint8_t, 64>& cuda_ipc_handle);

  /**
   * Close a previously opened CUDA IPC handle.
   * @param d_ptr Pointer returned by OpenHandleToCudaMemory.
   * @throws std::runtime_error if cudaIpcCloseMemHandle fails.
   */
  static void CloseHandleToCudaMemory(void* d_ptr);

  /**
   * Get Memory Info for current cuda process
   * @param free Pointer to free memory
   * @param total Pointer to total memory
   * @throws std::runtime_error if cudaMemGetInfo fails.
   */
  static void GetMemoryInfo(size_t* free, size_t* total);

  /**
 * Get the CUDA device ID from a GPU UUID string.
 * @param uuid_str GPU UUID string in the format "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
 * @return CUDA device ID
 * @throws std::runtime_error if no device with the given UUID is found.
 */
  static int GetDeviceIdFromUUID(const boost::uuids::uuid& gpu_uuid);
};

#endif // CUDA_UTILS_H