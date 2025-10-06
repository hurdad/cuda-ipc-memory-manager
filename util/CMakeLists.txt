find_package(Boost REQUIRED)

add_library(cuda-util STATIC
        CudaUtils.cu
        CudaUtils.h
        UUIDConverter.hpp
)

target_link_libraries(cuda-util)

target_include_directories(cuda-util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
