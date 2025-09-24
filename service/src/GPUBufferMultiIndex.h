#ifndef GPUBUFFERMULTIINDEX_H
#define GPUBUFFERMULTIINDEX_H

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/uuid/uuid.hpp>
#include <chrono>
#include <vector>

#include "common_generated.h" // fbs::cuda::ipc::api::ExpirationOption

/*
 * GPUBufferRecord
 * Represents a GPU buffer and metadata.
 * Optimized layout for 64-bit alignment.
 */
struct GPUBufferRecord {
  boost::uuids::uuid                    buffer_id; // 16 bytes
  void*                                 d_ptr = nullptr; // 8 bytes
  size_t                                size  = 0; // 8 bytes
  std::chrono::steady_clock::time_point creation_timestamp;
  std::chrono::steady_clock::time_point last_activity_timestamp;
  std::chrono::steady_clock::time_point expiration_timestamp; // precomputed
  fbs::cuda::ipc::api::CudaIPCHandle    ipc_handle;
  int32_t                               gpu_device_index = 0;
  fbs::cuda::ipc::api::ExpirationOption expiration_option;
  int32_t                               access_count = 0;
  std::vector<uint32_t>                 access_ids;
};

// Tags for Boost.MultiIndex
struct ByBufferId {
};

struct ByExpiration {
};

struct ByAccessOptionAndCount {
}; // Composite key: expiration_option.access_count + access_count

using namespace boost::multi_index;

/*
 * Composite key functor for ordering by:
 * 1. expiration_option.access_count() (primary)
 * 2. current access_count (secondary)
 * Compatible with Boost.MultiIndex by defining result_type and key_type.
 */
struct ExpAccessKey {
  using result_type = std::pair<uint64_t, uint64_t>;
  using key_type    = result_type;

  result_type operator()(const GPUBufferRecord& r) const {
    return std::make_pair(
        r.expiration_option.access_count(),
        static_cast<uint64_t>(r.access_count)
        );
  }
};

/*
 * GPUBufferMultiIndex
 * Boost.MultiIndex container supporting:
 * - Fast lookup by buffer_id
 * - Range queries by expiration_timestamp
 * - Composite ordering by allowed access count and current access count
 */
using GPUBufferMultiIndex = multi_index_container<
  GPUBufferRecord,
  indexed_by<
    // Unique index by buffer_id
    hashed_unique<
      tag<ByBufferId>,
      member<GPUBufferRecord, boost::uuids::uuid, &GPUBufferRecord::buffer_id>
    >,

    // Ordered index by expiration_timestamp
    ordered_non_unique<
      tag<ByExpiration>,
      member<GPUBufferRecord, std::chrono::steady_clock::time_point, &GPUBufferRecord::expiration_timestamp>
    >,

    // Composite index: expiration_option.access_count() + access_count
    ordered_non_unique<
      tag<ByAccessOptionAndCount>,
      ExpAccessKey
    >
  >
>;

#endif // GPUBUFFERMULTIINDEX_H