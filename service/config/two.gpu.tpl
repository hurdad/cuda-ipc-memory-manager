{
  "cuda_gpu_devices": [
    {
      "gpu_uuid": "${GPU_UUID_0}",
      "max_memory_allocation_type": "${MAX_MEMORY_ALLOCATION_TYPE_0:MaxGPUMemoryPercentage}",
      "max_memory_allocation": {
        "value": ${MAX_MEMORY_ALLOCATION_VALUE_0:1.0}
      }
    },
    {
      "gpu_uuid": "${GPU_UUID_1}",
      "max_memory_allocation_type": "${MAX_MEMORY_ALLOCATION_TYPE_1:MaxGPUMemoryPercentage}",
      "max_memory_allocation": {
        "value": ${MAX_MEMORY_ALLOCATION_VALUE_1:1.0}
      }
    }
  ],
  "zmq_request_endpoint": "${ZMQ_ROUTER_ENDPOINT:ipc:///tmp/cuda-ipc-memory-manager-service.ipc}",
  "prometheus_endpoint": "${PROMETHEUS_ENDPOINT:0.0.0.0:9242}",
  "expiration_thread_interval_ms": ${EXPIRATION_THREAD_INTERVAL_MS:1000}
}
