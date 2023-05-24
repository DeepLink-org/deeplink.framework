#pragma once

#include <cstring>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

  // using CambProcessGroupDICL = ProcessGroupDICL;

  // c10::intrusive_ptr<ProcessGroupDICL> createProcessGroupDICL(const c10::intrusive_ptr<c10d::Store> &store,
  //       int rank, int size, const std::chrono::duration<float> &timeout) {
  //   return c10::make_intrusive<CambProcessGroupDICL>(store, rank, size);
  // }

  // CNCL op mapping
  static std::map<ReduceOp::RedOpType, cnclReduceOp_t> cncl_op = {
      {ReduceOp::MIN, cnclMin},
      {ReduceOp::MAX, cnclMax},
      {ReduceOp::SUM, cnclSum},
      {ReduceOp::PRODUCT, cnclProd},
  };


  #define CNCL_THROW(cmd)                                              \
    do {                                                                    \
      cnclResult_t error = cmd;                                             \
      if (error != CNCL_RET_SUCCESS) {                                      \
        std::string err = "CNCL error in: " + std::string(__FILE__) + ":" + \
            std::to_string(__LINE__) + ", " +                               \
            std::string(cnclGetErrorStr(error));                            \
        throw std::runtime_error(err);                                      \
      }                                                                     \
    } while (0)

  #define CNCL_ASSERT(cmd)                                              \
    do {                                                                    \
      cnclResult_t res = cmd;                                             \
      if (res != CNCL_RET_SUCCESS) {                                      \
        std::string err = cnclGetErrorStr(res);    \
        fprintf(                                           \
            stderr,                                        \
            "CNCL error in: %s:%d, %s\n",                  \
            __FILE__,                                      \
            __LINE__,                                      \
            err.c_str());                                  \
        abort();                                           \
      }                                                                     \
    } while (0)

  #define DICL_RET(cmd)                                              \
    do {                                                                    \
      cnclResult_t error = cmd;                                             \
      if (error != CNCL_RET_SUCCESS) {                                      \
        std::string err = "CNCL error in: " + std::string(__FILE__) + ":" + \
            std::to_string(__LINE__) + ", " +                               \
            std::string(cnclGetErrorStr(error));                            \
        throw std::runtime_error(err);                                      \
      }                                                                     \
    } while (0)

}
}