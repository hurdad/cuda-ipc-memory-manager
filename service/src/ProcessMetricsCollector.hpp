#ifndef PROCESSMETRICSCOLLECTOR_HP
#define PROCESSMETRICSCOLLECTOR_HP

#include <prometheus/collectable.h>
#include <prometheus/metric_family.h>

#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <memory>

class ProcessMetricsCollector : public prometheus::Collectable {
public:
  std::vector<prometheus::MetricFamily> Collect() const override {
    std::vector<prometheus::MetricFamily> mfs;

    // ---------- CPU metric ----------
    prometheus::MetricFamily cpu_mf;
    cpu_mf.name = "process_cpu_seconds_total";
    cpu_mf.help = "Total user and system CPU time spent in seconds.";
    cpu_mf.type = prometheus::MetricType::Gauge; // FIXED

    prometheus::ClientMetric cpu_metric;
    cpu_metric.gauge.value = getProcessCPUSeconds();
    cpu_mf.metric.push_back(std::move(cpu_metric));
    mfs.push_back(std::move(cpu_mf));

    // ---------- Memory metric ----------
    prometheus::MetricFamily mem_mf;
    mem_mf.name = "process_memory_bytes";
    mem_mf.help = "Resident memory size in bytes.";
    mem_mf.type = prometheus::MetricType::Gauge; // FIXED

    prometheus::ClientMetric mem_metric;
    mem_metric.gauge.value = getProcessMemoryBytes();
    mem_mf.metric.push_back(std::move(mem_metric));
    mfs.push_back(std::move(mem_mf));

    return mfs;
  }

private:
  double getProcessCPUSeconds() const {
    std::ifstream stat("/proc/self/stat");
    std::string   line;
    std::getline(stat, line);
    if (!stat.good()) return 0.0;

    std::istringstream iss(line);
    std::string        token;
    int                field = 1;
    long               utime = 0, stime = 0;
    while (iss >> token) {
      if (field == 14) utime = std::stol(token);
      if (field == 15) {
        stime = std::stol(token);
        break;
      }
      ++field;
    }

    long ticks_per_sec = sysconf(_SC_CLK_TCK);
    return static_cast<double>(utime + stime) / ticks_per_sec;
  }

  double getProcessMemoryBytes() const {
    std::ifstream status("/proc/self/status");
    std::string   line;
    while (std::getline(status, line)) {
      if (line.rfind("VmRSS:", 0) == 0) {
        // Resident memory
        std::istringstream iss(line);
        std::string        key, unit;
        double             value_kb;
        iss >> key >> value_kb >> unit;
        return value_kb * 1024; // KB → Bytes
      }
    }
    return 0.0;
  }
};


#endif //PROCESSMETRICSCOLLECTOR_HP