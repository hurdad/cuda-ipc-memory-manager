#ifndef UUID_CONVERTER_H
#define UUID_CONVERTER_H

#include <cstdint>
#include <cstring>                     // For memcpy
#include <boost/uuid/uuid.hpp>         // For boost::uuids::uuid
#include "common_generated.h"          // For Flatbuffers UUID: fbs::cuda::ipc::api::UUID

namespace util {

class UUIDConverter {
public:
  /**
   * @brief Converts a boost::uuids::uuid to a Flatbuffers UUID.
   *
   * @param boostUuid The input boost UUID (16 bytes).
   * @return fbs::cuda::ipc::api::UUID The corresponding Flatbuffers UUID.
   *
   * @note Assumes platform endianness matches the expected Flatbuffers layout.
   */
  static fbs::cuda::ipc::api::UUID toFlatBufferUUID(const boost::uuids::uuid& boostUuid) noexcept {
    uint64_t uuid_h, uuid_l;
    std::memcpy(&uuid_h, boostUuid.data, sizeof(uint64_t));
    std::memcpy(&uuid_l, boostUuid.data + sizeof(uint64_t), sizeof(uint64_t));
    return fbs::cuda::ipc::api::UUID(uuid_l, uuid_h);
  }

  /**
   * @brief Converts a Flatbuffers UUID to a boost::uuids::uuid.
   *
   * @param flatBufferUuid The input Flatbuffers UUID.
   * @return boost::uuids::uuid The corresponding boost UUID.
   *
   * @note Assumes platform endianness matches the Flatbuffers layout.
   */
  static boost::uuids::uuid toBoostUUID(const fbs::cuda::ipc::api::UUID& flatBufferUuid) noexcept {
    boost::uuids::uuid boostUuid;
    uint64_t uuid_l = flatBufferUuid.lsb();
    uint64_t uuid_h = flatBufferUuid.msb();
    std::memcpy(boostUuid.data, &uuid_h, sizeof(uint64_t));
    std::memcpy(boostUuid.data + sizeof(uint64_t), &uuid_l, sizeof(uint64_t));
    return boostUuid;
  }
};

} // namespace util

#endif // UUID_CONVERTER_H
