#include "GpuMetricsCollector.h"
#include <prometheus/metric_family.h>
//#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "CudaUtils.h"

#define NVML_CHECK(call)                                             \
    do {                                                             \
        nvmlReturn_t result = call;                                  \
        if (result != NVML_SUCCESS) {                                \
            throw std::runtime_error(std::string("NVML Error: ") +   \
                                     nvmlErrorString(result));       \
        }                                                            \
    } while (0)

GpuMetricsCollector::GpuMetricsCollector() {
  NVML_CHECK(nvmlInit());
}

GpuMetricsCollector::~GpuMetricsCollector() {
  nvmlShutdown();
}

std::vector<prometheus::MetricFamily> GpuMetricsCollector::Collect() const {
  using namespace prometheus;
  std::vector<MetricFamily> mfs;

  unsigned int deviceCount = 0;
  NVML_CHECK(nvmlDeviceGetCount(&deviceCount));

  // Metric families
  MetricFamily gpuUtilFamily{"gpu_utilization_percent", {}, MetricType::Gauge};
  MetricFamily gpuMemUsedFamily{"gpu_memory_used_mib", {}, MetricType::Gauge};
  MetricFamily gpuMemTotalFamily{"gpu_memory_total_mib", {}, MetricType::Gauge};
  MetricFamily gpuMemUtilFamily{"gpu_memory_utilization_percent", {}, MetricType::Gauge};
  MetricFamily gpuMemFreeProcessFamily{"gpu_memory_free_process_mib", {}, MetricType::Gauge};
  MetricFamily gpuCudaTotalFamily{"gpu_memory_total_process_mib", {}, MetricType::Gauge};
  MetricFamily gpuTempFamily{"gpu_temperature_celsius", {}, MetricType::Gauge};
  MetricFamily gpuPowerFamily{"gpu_power_watts", {}, MetricType::Gauge};
  MetricFamily gpuFanFamily{"gpu_fan_percent", {}, MetricType::Gauge};

  for (unsigned int i = 0; i < deviceCount; ++i) {
    nvmlDevice_t device;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(i, &device));

    // NVML GPU utilization
    nvmlUtilization_t util;
    NVML_CHECK(nvmlDeviceGetUtilizationRates(device, &util));

    // NVML Temperature
    unsigned int temp;
    NVML_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp));

    // NVML Memory info
    nvmlMemory_t mem;
    NVML_CHECK(nvmlDeviceGetMemoryInfo(device, &mem));
    double memUsedMiB     = mem.used / 1024.0 / 1024.0;
    double memTotalMiB    = mem.total / 1024.0 / 1024.0;
    double memUtilPercent = (mem.total > 0) ? (100.0 * mem.used / mem.total) : 0.0;

    // NVML Power usage
    unsigned int power_mW;
    NVML_CHECK(nvmlDeviceGetPowerUsage(device, &power_mW));
    double power_W = power_mW / 1000.0;

    // NVML Fan speed
    unsigned int fanPercent   = 0;
    nvmlReturn_t fanRes       = nvmlDeviceGetFanSpeed(device, &fanPercent);
    bool         fanSupported = (fanRes == NVML_SUCCESS);

    // CUDA memory info for this process and device
    size_t freeMem = 0, totalMem = 0;
    CudaUtils::SetDevice(i);
    CudaUtils::GetMemoryInfo(&freeMem, &totalMem);
    double freeMemMiB   = freeMem / 1024.0 / 1024.0;
    double cudaTotalMiB = totalMem / 1024.0 / 1024.0;

    // Prepare labels vector
    std::vector<ClientMetric::Label> labels{{"gpu", std::to_string(i)}};

    ClientMetric cm;

    // GPU utilization %
    cm.label       = labels;
    cm.gauge.value = static_cast<double>(util.gpu);
    gpuUtilFamily.metric.push_back(cm);

    // Memory utilization %
    cm.gauge.value = memUtilPercent;
    gpuMemUtilFamily.metric.push_back(cm);

    // Memory used MiB
    cm.gauge.value = memUsedMiB;
    gpuMemUsedFamily.metric.push_back(cm);

    // Memory total MiB
    cm.gauge.value = memTotalMiB;
    gpuMemTotalFamily.metric.push_back(cm);

    // Free process memory
    cm.gauge.value = freeMemMiB;
    gpuMemFreeProcessFamily.metric.push_back(cm);

    // Total CUDA process memory
    cm.gauge.value = cudaTotalMiB;
    gpuCudaTotalFamily.metric.push_back(cm);

    // GPU temperature
    cm.gauge.value = static_cast<double>(temp);
    gpuTempFamily.metric.push_back(cm);

    // Power usage
    cm.gauge.value = power_W;
    gpuPowerFamily.metric.push_back(cm);

    // Fan speed
    if (fanSupported) {
      cm.gauge.value = static_cast<double>(fanPercent);
      gpuFanFamily.metric.push_back(cm);
    }
  }

  // Add MetricFamilies to output
  mfs.push_back(gpuUtilFamily);
  mfs.push_back(gpuMemUsedFamily);
  mfs.push_back(gpuMemTotalFamily);
  mfs.push_back(gpuMemUtilFamily);
  mfs.push_back(gpuMemFreeProcessFamily);
  mfs.push_back(gpuCudaTotalFamily);
  mfs.push_back(gpuTempFamily);
  mfs.push_back(gpuPowerFamily);
  if (!gpuFanFamily.metric.empty())
    mfs.push_back(gpuFanFamily);

  return mfs;
}