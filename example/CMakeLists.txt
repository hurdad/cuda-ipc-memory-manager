find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(create-buffer-example
       CreateBufferExample.cu
)

target_include_directories(create-buffer-example
        PRIVATE
        ${CMAKE_SOURCE_DIR}/schema/external/flatbuffers/include
)

target_link_libraries(create-buffer-example PRIVATE
        cuda-ipc-memory-manager-api
)

add_executable(get-buffer-example
        GetBufferExample.cu
)

target_include_directories(get-buffer-example
        PRIVATE
        ${CMAKE_SOURCE_DIR}/schema/external/flatbuffers/include
)

target_link_libraries(get-buffer-example PRIVATE
        cuda-ipc-memory-manager-api
)

add_executable(free-buffer-example
        FreeBufferExample.cu
)

target_include_directories(free-buffer-example
        PRIVATE
        ${CMAKE_SOURCE_DIR}/schema/external/flatbuffers/include
)

target_link_libraries(free-buffer-example PRIVATE
        cuda-ipc-memory-manager-api
)
