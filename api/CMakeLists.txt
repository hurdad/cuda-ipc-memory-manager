find_package(spdlog REQUIRED)

add_library(cuda-ipc-memory-manager-api STATIC
        API.cpp
       API.h
)

target_include_directories(cuda-ipc-memory-manager-api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cuda-ipc-memory-manager-api PUBLIC etcd flatbuffers_generated spdlog::spdlog)
