find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

set(ENABLE_TESTING OFF CACHE BOOL "" FORCE) # Disable prometheus-cpp tests
set(ENABLE_PUSH OFF CACHE BOOL "" FORCE) # Disable building the prometheus-cpp push lib
add_subdirectory(external/prometheus-cpp)

add_executable(cuda-ipc-memory-manager-service
        src/Main.cpp
        src/ConfigParser.hpp
        src/CudaIPCServer.cpp
        src/CudaIPCServer.h
        src/GpuMetricsCollector.cpp
        src/GpuMetricsCollector.h
        src/GPUBufferMultiIndex.h
        src/ProcessMetricsCollector.hpp
)

target_include_directories(cuda-ipc-memory-manager-service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

option(USE_NVML "Enable NVML support" ON)
if(USE_NVML)
    message(STATUS "Using NVML")
    target_compile_definitions(cuda-ipc-memory-manager-service PRIVATE HAVE_NVML)

    target_link_libraries(cuda-ipc-memory-manager-service PRIVATE
            Boost::program_options
            flatbuffers
            cuda-ipc-flatbuffers-generated
            zmq
            spdlog::spdlog
            cuda-util
            prometheus-cpp::pull
            nvidia-ml
    )
else()
    message(WARNING "Disabling NVML support.")
    set(USE_NVML OFF)

    target_link_libraries(cuda-ipc-memory-manager-service PRIVATE
            Boost::program_options
            flatbuffers
            cuda-ipc-flatbuffers-generated
            zmq
            spdlog::spdlog
            cuda-util
            prometheus-cpp::pull
    )
endif()
