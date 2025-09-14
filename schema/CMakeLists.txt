# --- FlatBuffers codegen setup ---

# Directories
set(FBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GEN_DIR}")

# Make sure we have the flatbuffers compiler `flatc`
find_program(FLATC_EXECUTABLE flatc REQUIRED)

# Recursively find all .fbs files in the current directory
file(GLOB_RECURSE FBS_SRCS "${FBS_DIR}/*.fbs")

message(STATUS "Found flatbuffers files:")
foreach (fbs_file ${FBS_SRCS})
    message(STATUS "  ${fbs_file}")
endforeach ()

# Collect generated headers + include dirs
set(GEN_HDRS "")
set(GEN_INCLUDE_DIRS "")

foreach (fbs_file ${FBS_SRCS})
    # Path relative to schema root
    file(RELATIVE_PATH rel_path "${FBS_DIR}" "${fbs_file}")
    get_filename_component(fbs_name "${rel_path}" NAME_WE)
    get_filename_component(fbs_dir  "${rel_path}" PATH)

    # Directory where generated file will go
    set(gen_dir_full "${GEN_DIR}/${fbs_dir}")
    file(MAKE_DIRECTORY "${gen_dir_full}")

    # Expected generated header
    set(gen_hdr "${gen_dir_full}/${fbs_name}_generated.h")
    list(APPEND GEN_HDRS ${gen_hdr})

    # Add include directory for this schema folder
    list(APPEND GEN_INCLUDE_DIRS "${gen_dir_full}")

    # Rule to generate it
    add_custom_command(
            OUTPUT "${gen_hdr}"
            COMMAND ${FLATC_EXECUTABLE}
            --cpp
            --bfbs-gen-embed
            -I "${FBS_DIR}"
            -o "${gen_dir_full}"
            "${fbs_file}"
            DEPENDS "${fbs_file}"
            COMMENT "Generating Flatbuffers C++ header for ${fbs_file}"
            VERBATIM
    )
endforeach ()

list(REMOVE_DUPLICATES GEN_INCLUDE_DIRS)

# Custom target to group all generation
add_custom_target(generate_fbs DEPENDS ${GEN_HDRS})

# Interface library to propagate includes + build order
add_library(flatbuffers_generated INTERFACE)
add_dependencies(flatbuffers_generated generate_fbs)
target_include_directories(flatbuffers_generated INTERFACE ${GEN_INCLUDE_DIRS})

