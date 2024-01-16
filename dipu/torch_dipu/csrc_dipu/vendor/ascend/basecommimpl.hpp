#pragma once

#include <acl/acl.h>
#include <cstring>
#include <unistd.h>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

template <typename Key, typename Value, std::size_t Size>
struct Map {
  std::array<std::pair<Key, Value>, Size> data;

  [[nodiscard]] constexpr Value at(const Key& key) const {
    const auto itr =
        std::find_if(begin(data), end(data),
                     [&key](const auto& v) { return v.first == key; });
    if (itr != end(data)) {
      return itr->second;
    } else {
      TORCH_CHECK(false, "Not Found");
    }
  }
};

// HCCL ReduceOp mapping
std::map<c10d::ReduceOp, HcclReduceOp> hcclOp = {
    {ReduceOp::MIN, HCCL_REDUCE_MIN},
    {ReduceOp::MAX, HCCL_REDUCE_MAX},
    {ReduceOp::SUM, HCCL_REDUCE_SUM},
    {ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
};

bool isPinnedPtr(const void* p) {
  TORCH_CHECK(false, "isPinnedPtr not implemented for ascend.\n");
  return false;
}

#define TRACK_FUN_CALL(TAG, x)                                       \
  {                                                                  \
    static bool enable = std::getenv("DIPU_TRACK_" #TAG) != nullptr; \
    if (enable) {                                                    \
      printf("[%d %s: %d]:%s\n", getpid(), __FILE__, __LINE__, x);   \
    }                                                                \
  }

#define DIPU_CALLACLRT(Expr)                                               \
  {                                                                        \
    TRACK_FUN_CALL(ACL, #Expr);                                            \
    ::aclError ret = Expr;                                                 \
    TORCH_CHECK(ret == ACL_SUCCESS, "ascend device error, expr = ", #Expr, \
                ", ret = ", ret, ", error msg = ", aclGetRecentErrMsg());  \
  }

#define HCCL_THROW(cmd)                                           \
  do {                                                            \
    TRACK_FUN_CALL(HCCL, #cmd)                                    \
    TORCH_CHECK(cmd == HCCL_SUCCESS,                              \
                "HCCL error in: " + std::string(__FILE__) + ":" + \
                    std::to_string(__LINE__) + ".\n" +            \
                    "And see details in Ascend logs.\n" +         \
                    aclGetRecentErrMsg());                        \
  } while (0)

}  // namespace devapis
}  // namespace dipu
