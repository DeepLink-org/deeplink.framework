#pragma once

#include <cstring>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

// ECCL op mapping
static std::map<ReduceOp::RedOpType, ecclRedOp_t> eccl_op = {
    {ReduceOp::MIN, ecclMin},
    {ReduceOp::MAX, ecclMax},
    {ReduceOp::SUM, ecclSum},
    {ReduceOp::PRODUCT, ecclProd},
};

#define ECCL_THROW(cmd)                                                   \
  do {                                                                    \
    ecclResult_t error = cmd;                                             \
    if (error != ecclSuccess) {                                           \
      std::string err = "ECCL error in: " + std::string(__FILE__) + ":" + \
                        std::to_string(__LINE__) + ", " +                 \
                        std::string(ecclGetErrorString(error));           \
      throw std::runtime_error(err);                                      \
    }                                                                     \
  } while (0)

#define ECCL_ASSERT(cmd)                                                \
  do {                                                                  \
    ecclResult_t res = cmd;                                             \
    if (res != ecclSuccess) {                                           \
      std::string err = ecclGetErrorString(res);                        \
      fprintf(stderr, "ECCL error in: %s:%d, %s\n", __FILE__, __LINE__, \
              err.c_str());                                             \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define ECCL_RET(cmd)                                                     \
  do {                                                                    \
    ecclResult_t error = cmd;                                             \
    if (error != ecclSuccess) {                                           \
      std::string err = "ECCL error in: " + std::string(__FILE__) + ":" + \
                        std::to_string(__LINE__) + ", " +                 \
                        std::string(ecclGetErrorString(error));           \
      throw std::runtime_error(err);                                      \
    }                                                                     \
  } while (0)

}  // namespace devapis
}  // namespace dipu