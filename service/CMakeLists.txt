find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(cuda-ipc-memory-manager-service
        src/Main.cpp
        src/ConfigParser.hpp
        src/thread/CudaIPCServer.cpp
        src/thread/CudaIPCServer.h
        src/thread/CudaIPCServer.h
        src/thread/UUIDConverter.hpp
        src/thread/CudaUtils.cu
        src/thread/CudaUtils.h
)

target_link_libraries(cuda-ipc-memory-manager-service PRIVATE
        Boost::program_options
        flatbuffers
        flatbuffers_generated
        zmq
        spdlog::spdlog
)

set_target_properties(cuda-ipc-memory-manager-service PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
)

target_compile_options(cuda-ipc-memory-manager-service PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)

target_include_directories(cuda-ipc-memory-manager-service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
