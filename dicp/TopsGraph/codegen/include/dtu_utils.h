#pragma once

#include <memory>
#include <vector>

#include "tops/tops_ext.h"
#include "tops/tops_runtime.h"
#include "dtu_compiler/tops_graph_compiler.h"
#include "hlir_builder/hlir_builder.h"

#define MAX_NUM 4096

#define EXPECT_EQ(_src, _dst)                                                  \
  do {                                                                         \
    if ((_src) != (_dst)) {                                                    \
      printf("FAIL: %s:%d,%s() %s != %s\n", __FILE__, __LINE__, __func__,      \
             #_src, #_dst);                                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define EXPECT_NE(_src, _dst)                                                  \
  do {                                                                         \
    if ((_src) == (_dst)) {                                                    \
      printf("FAIL: %s:%d,%s() %s == %s\n", __FILE__, __LINE__, __func__,      \
             #_src, #_dst);                                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

bool file_exists(const char *filename);
void compile(
    std::shared_ptr<builder::Builder> builder,
    topsExecutable_t* exe_ptr);

int run(
    topsExecutable_t exe_ptr,
    std::vector<void*>& input_ptrs,
    std::vector<void*>& output_ptrs,
    int device_id);

int runV2(
    topsExecutable_t exe_ptr,
    std::vector<void*>& input_ptrs,
    size_t* input_dims,
    size_t* input_rank,
    std::vector<void*>& output_ptrs,
    size_t* output_dims,
    size_t* output_rank);

