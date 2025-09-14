find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
add_subdirectory(external/prometheus-cpp)

add_executable(cuda-ipc-memory-manager-service
        src/Main.cpp
        src/ConfigParser.hpp
        src/thread/CudaIPCServer.cpp
        src/thread/CudaIPCServer.h
)

target_link_libraries(cuda-ipc-memory-manager-service PRIVATE
        Boost::program_options
        flatbuffers
        flatbuffers_generated
        zmq
        spdlog::spdlog
        cuda-util
        prometheus-cpp::pull
)

target_include_directories(cuda-ipc-memory-manager-service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
