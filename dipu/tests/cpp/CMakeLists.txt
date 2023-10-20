file(GLOB test1 testrt.cpp)

# temporarily: enhance to use the same configuration as lib
set(DIPU_PATH ${PROJECT_SOURCE_DIR}/torch_dipu/)
include_directories(${DIPU_PATH})
include_directories(${DIPU_PATH}/dist/include)
add_subdirectory(${DIPU_PATH}/csrc_dipu/vendor/${UsedVendor} vendor_bins_dir)
include_directories(SYSTEM ${VENDOR_INCLUDE_DIRS})
link_directories(${VENDOR_LIB_DIRS})

# to use gtest
set(ALL_TESTS test_tensor_add test_relu testrt)
foreach(tname ${ALL_TESTS})
  add_executable(${tname} ${tname}.cpp)
  target_link_libraries(${tname} torch_dipu)
  target_link_libraries(${tname} c10 torch torch_cpu)
endforeach(tname)
