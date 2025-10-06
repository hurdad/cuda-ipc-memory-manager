find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)


add_executable(get-info-example
        GetInfoExample.cpp
)

target_link_libraries(get-info-example PRIVATE
        cuda-ipc-memory-manager-api
)

add_executable(create-buffer-example
       CreateBufferExample.cu
)

target_link_libraries(create-buffer-example PRIVATE
        cuda-ipc-memory-manager-api
)

add_executable(get-buffer-example
        GetBufferExample.cu
)

target_link_libraries(get-buffer-example PRIVATE
        cuda-ipc-memory-manager-api
)

add_executable(free-buffer-example
        FreeBufferExample.cpp
)

target_link_libraries(free-buffer-example PRIVATE
        cuda-ipc-memory-manager-api
)
