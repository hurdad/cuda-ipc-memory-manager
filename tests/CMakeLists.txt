# Find GoogleTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include the main project headers
include_directories(..)

# List all test source files
set(TEST_SOURCES
        CudaIpcMemoryManagerAPITest.cpp
)

# Create a test executable for each test file
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE cuda-ipc-memory-manager-api GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()