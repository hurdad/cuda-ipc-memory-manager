#ifndef GPUMETRICSCOLLECTOR_H
#define GPUMETRICSCOLLECTOR_H

#include <nvml.h>
#include <prometheus/collectable.h>

#include <boost/uuid/uuid_generators.hpp>
#include <vector>

#include "service_generated.h"

class GpuMetricsCollector : public prometheus::Collectable {
 public:
  GpuMetricsCollector(const fbs::cuda::ipc::service::Configuration* configuration);
  ~GpuMetricsCollector() override;

  std::vector<prometheus::MetricFamily> Collect() const override;

 private:
  const fbs::cuda::ipc::service::Configuration* configuration_;
};

#endif // GPUMETRICSCOLLECTOR_H