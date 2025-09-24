#ifndef GPUBUFFERMULTIINDEX_H
#define GPUBUFFERMULTIINDEX_H

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/uuid/uuid.hpp>
#include <chrono>
#include <vector>

struct GPUBufferRecord {
  boost::uuids::uuid                     buffer_id;
  fbs::cuda::ipc::api::CudaIPCHandle     ipc_handle;
  int                                    gpu_device_index  = 0;
  fbs::cuda::ipc::api::ExpirationOptions expiration_option = fbs::cuda::ipc::api::ExpirationOptions_NONE;
  void*                                  d_ptr             = nullptr;
  size_t                                 size              = 0;
  std::chrono::steady_clock::time_point  creation_timestamp;
  std::chrono::steady_clock::time_point  last_activity_timestamp;
  std::chrono::steady_clock::time_point  expiration_timestamp;
  std::vector<uint32_t>                  access_ids;
};

#include <boost/multi_index/member.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>

using namespace boost::multi_index;

// Tags for indices
struct ByBufferId {
};

struct ByExpiration {
};

// Define multi-index container
using GPUBufferMultiIndex = multi_index_container<
  GPUBufferRecord,
  indexed_by<
    // Index by buffer_id (hashed for fast lookup)
    hashed_unique<
      tag<ByBufferId>,
      member<GPUBufferRecord, boost::uuids::uuid, &GPUBufferRecord::buffer_id>
    >,
    // Index by expiration timestamp (ordered for range queries)
    ordered_non_unique<
      tag<ByExpiration>,
      member<GPUBufferRecord, std::chrono::steady_clock::time_point, &GPUBufferRecord::expiration_timestamp>
    >
  >
>;

#endif //GPUBUFFERMULTIINDEX_H