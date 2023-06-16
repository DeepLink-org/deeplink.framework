
MESSAGE(STATUS "STPU_INSTALL = ${STPU_INSTALL}.")

find_program(STPU_CLANG_EXECUTABLE
    NAMES clang++
    HINTS ${STPU_INSTALL}/bin
)

set(CMAKE_STPU_CLANG_FLAGS "${CMAKE_STPU_CLANG_FLAGS};-std=c++14;-fPIC;-O2;")
MESSAGE(STATUS "STPU_CLANG_EXECUTABLE = ${STPU_CLANG_EXECUTABLE}.")

macro(STPU_COMPILE generated_files)
    foreach(src_file ${ARGN})
        set(file "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
        get_filename_component(basename ${file} NAME)
        set(output_file_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${basename}.stpu.dir")
        set(compiled_file "${output_file_dir}/${basename}.o")

        add_custom_command(
            OUTPUT ${compiled_file}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${output_file_dir}"
            COMMAND ${STPU_CLANG_EXECUTABLE} ${CMAKE_STPU_CLANG_FLAGS}
                    "-o" ${compiled_file}
                    "-c" ${file}
                    "-I" ${STPU_INSTALL}/include
            WORKING_DIRECTORY ${output_file_dir}
        )
        add_custom_target(${basename}.stpu DEPENDS ${compiled_file})
        list(APPEND ${generated_files} ${compiled_file})
    endforeach()
endmacro()
