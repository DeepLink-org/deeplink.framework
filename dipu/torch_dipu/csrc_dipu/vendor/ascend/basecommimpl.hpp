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


#define TRACK_HCCL(x)                                                     \
  {                                                                   \
    static bool enable = std::getenv("DIPU_TRACK_HCCL") != nullptr; \
    if (enable) {                                                   \
      printf("[%d %s: %d]:%s\n", getpid(), __FUNCTION__, __LINE__, x);           \
    }                                                               \
  }

#define HCCL_THROW(cmd)                                           \
  do {                                                            \
    TRACK_HCCL(#cmd)                                              \
    TORCH_CHECK(cmd == HCCL_SUCCESS,                              \
                "HCCL error in: " + std::string(__FILE__) + ":" + \
                    std::to_string(__LINE__) + ".\n" +            \
                    "And see details in Ascend logs.\n" +         \
                    aclGetRecentErrMsg());                        \
  } while (0)

}  // namespace devapis
}  // namespace dipu
