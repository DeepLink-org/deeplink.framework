function(git_commit_id output_variable)
  find_package(Git)

  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" describe --match='' --always --abbrev=40 --dirty
      OUTPUT_VARIABLE OUTPUT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  else()
    set(OUTPUT "unknown")
    message(STATUS "Cannot find git tools, fallback to 'unknown'")
  endif()

  set(${output_variable} "${OUTPUT}" PARENT_SCOPE)
endfunction()

function(version_to_number version output_variable)
  string(REPLACE "." ";" PARTS "${version}")
  list(POP_FRONT PARTS NUMBER)

  foreach(PART ${PARTS})
    if(PART MATCHES "^[0-9][0-9]?$")
      string(REGEX REPLACE "^.*(..)$" "\\1" PART "00${PART}")
      string(APPEND NUMBER ${PART})
    else()
      message(FATAL_ERROR "Version number '${PART}' in ${version} is not supported.")
    endif()
  endforeach()

  set(${output_variable} ${NUMBER} PARENT_SCOPE)
endfunction()

function(detect_abi_versin output_variable)
  execute_process(
    COMMAND
      "${Python3_EXECUTABLE}" -c "\
import torch, builtins\n\
from pathlib import Path\n\
print(next(item[-4:-2] for item in dir(builtins) if \"__pybind11_internals_v4_gcc_libstdcpp_cxxabi10\" in item))"
    OUTPUT_VARIABLE OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  set(${output_variable} ${OUTPUT} PARENT_SCOPE)
endfunction()

function(detect_device_name hint output_variable)
  set(DEVICE_SET_0 cuda CUDA)
  set(DEVICE_SET_1 camb CAMB)
  set(DEVICE_SET_2 ascend ASCEND)
  set(DEVICE_SET_3 topsrider TOPSRIDER TOPS tops)
  set(DEVICE_SET_4 supa SUPA)
  set(DEVICE_SET_5 droplet DROPLET)
  set(DEVICE_SET_6 kunlunxin klx)

  foreach(X RANGE 6) # [0, 6], 6 is included.
    if(${hint} IN_LIST DEVICE_SET_${X})
      list(GET DEVICE_SET_${X} 0 OUTPUT)
      set(${output_variable} ${OUTPUT} PARENT_SCOPE)
      return()
    endif()
  endforeach()

  message(FATAL_ERROR "Fail to detect device name form '${hint}'.")
endfunction()

function(device_name_to_opt_name device_name output_variable)
  set(SOURCE cuda topsrider)
  set(TARGET torch tops)

  foreach(KEY VALUE IN ZIP_LISTS SOURCE TARGET)
    if(KEY STREQUAL device_name)
      set(${output_variable} ${VALUE} PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${output_variable} ${device_name} PARENT_SCOPE)
endfunction()

function(generate_supported_diopi_functions input_file output_file)
  file(STRINGS "${input_file}" FUNCTION_NAMES REGEX "diopi[0-9a-zA-Z]+\\(")
  list(TRANSFORM FUNCTION_NAMES REPLACE ".*(diopi[0-9a-zA-Z]+)\\(.+" "\\1\n")
  list(SORT FUNCTION_NAMES CASE INSENSITIVE)
  list(REMOVE_DUPLICATES FUNCTION_NAMES)
  file(WRITE "${output_file}" ${FUNCTION_NAMES})
endfunction()

function(_set_cpp_flags)
  # symbol hidden, cannot open now
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)

  # open flags cause many prpblem, fix return-type err and re-close
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" PARENT_SCOPE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-type-limits")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-overflow")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-qualifiers")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=redundant-decls")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=old-style-cast")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wreturn-type")

  set(CMAKE_CXX_FLAGS_RELEASE "-O2")

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -O0 -g")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
    set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
    set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CXXFLAGS}")
  else()
    set(CMAKE_C_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-s,-z,noexecstack -fPIE -pie ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-s,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
    set(CXXFLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-s,-z,noexecstack -fPIE -pie ${CXXFLAGS}")
  endif()
endfunction(_set_cpp_flags)
