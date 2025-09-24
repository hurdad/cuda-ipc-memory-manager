find_package(spdlog REQUIRED)
find_package(Boost REQUIRED)

add_library(cuda-ipc-memory-manager-api STATIC
        CudaIpcMemoryManagerAPI.h
        CudaIpcMemoryManagerAPI.cpp
)

target_link_libraries(cuda-ipc-memory-manager-api PUBLIC
        cuda-util
        cuda-ipc-flatbuffers-generated
        zmq
        spdlog::spdlog)

target_include_directories(cuda-ipc-memory-manager-api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})