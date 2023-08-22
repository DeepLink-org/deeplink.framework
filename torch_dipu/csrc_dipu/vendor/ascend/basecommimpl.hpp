#pragma once

#include <acl/acl.h>
#include <cstring>
#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

template <typename Key, typename Value, std::size_t Size>
struct Map {
  std::array<std::pair<Key, Value>, Size> data;

  [[nodiscard]] constexpr Value at(const Key &key) const {
    const auto itr =
        std::find_if(begin(data), end(data),
                     [&key](const auto &v) { return v.first == key; });
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

bool isPinnedPtr(const void *p)
{
  TORCH_CHECK(false, "isPinnedPtr not implemented for ascend.\n");     
  return false;
}

#define HCCL_THROW(cmd)                                                   \
  do {                                                                    \
    TORCH_CHECK(cmd == HCCL_SUCCESS, "HCCL error in: " +                  \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +  \
                ".\n" + "And see details in Ascend logs.\n" +             \
                aclGetRecentErrMsg());                                    \
  } while (0)

} //  ns devapis
} //  ns dipu