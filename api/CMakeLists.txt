find_package(spdlog REQUIRED)
find_package(Boost REQUIRED)

add_library(cuda-ipc-memory-manager-api STATIC
        CudaIpcMemoryRequestAPI.h
        CudaIpcMemoryRequestAPI.cpp
)

target_link_libraries(cuda-ipc-memory-manager-api PUBLIC
        cuda-util
        flatbuffers_generated
        zmq
        spdlog::spdlog)
