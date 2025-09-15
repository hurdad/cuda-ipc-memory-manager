find_package(spdlog REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(create-buffer-example
       CreateBufferExample.cpp
)

target_link_libraries(create-buffer-example PRIVATE
        cuda-ipc-memory-manager-api

)

add_executable(get-buffer-example
        GetBufferExample.cpp
)

target_link_libraries(get-buffer-example PRIVATE
        cuda-ipc-memory-manager-api
)
