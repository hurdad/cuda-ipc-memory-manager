#include "GpuMetricsCollector.h"

#include <prometheus/metric_family.h>
// #include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "CudaUtils.h"

#define NVML_CHECK(call)                                                               \
  do {                                                                                 \
    nvmlReturn_t result = call;                                                        \
    if (result != NVML_SUCCESS) {                                                      \
      throw std::runtime_error(std::string("NVML Error: ") + nvmlErrorString(result)); \
    }                                                                                  \
  } while (0)

GpuMetricsCollector::GpuMetricsCollector(const fbs::cuda::ipc::service::Configuration* configuration) : configuration_(configuration) {
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
  MetricFamily gpuInfoFamily{"nvidia_gpu_info", "A metric with a constant '1' value labeled by gpu uuid, name, driver_version.", MetricType::Gauge};
  MetricFamily gpuUtilFamily{"nvidia_gpu_utilization_percent", "Current utilization percentage of the NVIDIA GPU", MetricType::Gauge};
  MetricFamily gpuMemUsedFamily{"nvidia_gpu_memory_used_bytes", "Amount of GPU memory currently used in bytes", MetricType::Gauge};
  MetricFamily gpuMemFreeFamily{"nvidia_gpu_memory_free_bytes", "Amount of free GPU memory in bytes", MetricType::Gauge};
  MetricFamily gpuMemTotalFamily{"nvidia_gpu_memory_total_bytes", "Total GPU memory available in bytes", MetricType::Gauge};
  MetricFamily gpuMemUtilFamily{"nvidia_gpu_memory_utilization_percent", "Current GPU memory utilization percentage", MetricType::Gauge};
  MetricFamily gpuTempFamily{"nvidia_gpu_temperature_celsius", "Current temperature of the GPU in Celsius", MetricType::Gauge};
  MetricFamily gpuPowerFamily{"nvidia_gpu_power_watts", "Current power consumption of the GPU in watts", MetricType::Gauge};
  MetricFamily gpuFanFamily{"nvidia_gpu_fan_percent", "Current fan speed percentage of the GPU", MetricType::Gauge};

  // CUDA memory per process
  MetricFamily gpuCudaMemFreeProcessFamily{"cuda_ipc_gpu_memory_free_bytes", "Amount of free GPU memory available to this CUDA process in bytes",
                                           MetricType::Gauge};
  MetricFamily gpuCudaMemTotalProcessFamily{"cuda_ipc_gpu_memory_total_bytes", "Total GPU memory available to this CUDA process in bytes",
                                            MetricType::Gauge};

  // loop configured cuda gpu devices
  auto cuda_gpu_devices = configuration_->cuda_gpu_devices();
  if (cuda_gpu_devices) {
    for (auto cuda_gpu_device : *cuda_gpu_devices) {
      // convert gpu_uuid to boost uuid
      boost::uuids::string_generator gen;
      auto                           boost_gpu_uuid = gen(cuda_gpu_device->gpu_uuid()->str());

      // get device handle by uuid
      nvmlDevice_t device;
      NVML_CHECK(nvmlDeviceGetHandleByUUID(cuda_gpu_device->gpu_uuid()->c_str(), &device));

      // NVML GPU Name
      char name[NVML_DEVICE_NAME_BUFFER_SIZE];
      NVML_CHECK(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));

      // Get driver version
      char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
      NVML_CHECK(nvmlSystemGetDriverVersion(driver_version, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));

      // NVML GPU utilization
      nvmlUtilization_t util;
      NVML_CHECK(nvmlDeviceGetUtilizationRates(device, &util));

      // NVML Temperature
      unsigned int temp;
      NVML_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp));

      // NVML Memory info
      nvmlMemory_t mem;
      NVML_CHECK(nvmlDeviceGetMemoryInfo(device, &mem));
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
      int    cuda_device_id = CudaUtils::GetDeviceIdFromUUID(boost_gpu_uuid);
      CudaUtils::SetDevice(cuda_device_id);
      CudaUtils::GetMemoryInfo(&freeMem, &totalMem);

      // GPU info
      std::vector<ClientMetric::Label> info_labels{{"uuid", cuda_gpu_device->gpu_uuid()->str()}, {"name", name}, {"driver_version", driver_version}};
      ClientMetric          cm_info;
      cm_info.label       = info_labels;
      cm_info.gauge.value = 1;
      gpuInfoFamily.metric.push_back(cm_info);

      // Prepare labels vector
      std::vector<ClientMetric::Label> labels{{"uuid", cuda_gpu_device->gpu_uuid()->str()}};
      ClientMetric                     cm;
      cm.label = labels;

      // GPU utilization %
      cm.gauge.value = static_cast<double>(util.gpu);
      gpuUtilFamily.metric.push_back(cm);

      // Memory utilization %
      cm.gauge.value = memUtilPercent;
      gpuMemUtilFamily.metric.push_back(cm);

      // Memory used
      cm.gauge.value = mem.used;
      gpuMemUsedFamily.metric.push_back(cm);

      // Memory free
      cm.gauge.value = mem.free;
      gpuMemFreeFamily.metric.push_back(cm);

      // Memory total
      cm.gauge.value = mem.total;
      gpuMemTotalFamily.metric.push_back(cm);

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

      // Free CUDA process memory
      cm.gauge.value = freeMem;
      gpuCudaMemFreeProcessFamily.metric.push_back(cm);

      // Total CUDA process memory
      cm.gauge.value = totalMem;
      gpuCudaMemTotalProcessFamily.metric.push_back(cm);
    }
  }

  // Add MetricFamilies to output
  mfs.push_back(gpuInfoFamily);
  mfs.push_back(gpuUtilFamily);
  mfs.push_back(gpuMemUsedFamily);
  mfs.push_back(gpuMemFreeFamily);
  mfs.push_back(gpuMemTotalFamily);
  mfs.push_back(gpuMemUtilFamily);
  mfs.push_back(gpuTempFamily);
  mfs.push_back(gpuPowerFamily);
  if (!gpuFanFamily.metric.empty()) mfs.push_back(gpuFanFamily);
  mfs.push_back(gpuCudaMemFreeProcessFamily);
  mfs.push_back(gpuCudaMemTotalProcessFamily);
  return mfs;
}