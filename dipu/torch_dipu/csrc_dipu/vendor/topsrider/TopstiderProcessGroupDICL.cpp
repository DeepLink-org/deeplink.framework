// Copyright (c) 2024, DeepLink.

#include "csrc_dipu/runtime/distributed/ProcessGroupDICL.h"

namespace dipu {

std::unique_ptr<AllReduceStrategy> createAllReduceStrategy() {
  return std::make_unique<AllReduceStrategy>();
}

}  // namespace dipu
