{
  "cuda_gpu_devices": [
    {
      "cuda_gpu_index": 0,
      "max_memory_allocation_type": "{MAX_MEMORY_ALLOCATION_TYPE:MaxGPUMemoryPercentage}",
      "max_memory_allocation": {
        "value": {MAX_MEMORY_ALLOCATION_VALUE}"
      }
    }
  ],
  "zmq_request_endpoint": "${ZMQ_ROUTER_ENDPOINT:ipc:///tmp/cuda-ipc-memory-manager-service.ipc}",
  "prometheus_endpoint": "${PROMETHEUS_ENDPOINT:0.0.0.0:9242}"
  "expiration_thread_interval_ms": ${EXPIRATION_THREAD_INTERVAL_MS:0.0.0.0:9242}
}
