#ifndef GPUMETRICSCOLLECTOR_H
#define GPUMETRICSCOLLECTOR_H

#include <prometheus/collectable.h>
#include <vector>
#include <nvml.h>

class GpuMetricsCollector : public prometheus::Collectable {
public:
  GpuMetricsCollector();
  ~GpuMetricsCollector() override;

  std::vector<prometheus::MetricFamily> Collect() const override;
};


#endif //GPUMETRICSCOLLECTOR_H