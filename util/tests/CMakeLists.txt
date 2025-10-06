
# Enable testing
enable_testing()

# Find GoogleTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Include the main project headers
include_directories(..)

# List all test source files
set(TEST_SOURCES
    CudaUtilsTest.cpp
    UUIDConverterTest.cpp
)

# Create a test executable for each test file
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE cuda-util cuda-ipc-flatbuffers-generated GTest::gtest_main)
    target_include_directories(${test_name}
            PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/../../schema/external/flatbuffers/include
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()