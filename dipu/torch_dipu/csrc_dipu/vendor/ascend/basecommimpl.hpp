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

}  // namespace devapis
}  // namespace dipu
