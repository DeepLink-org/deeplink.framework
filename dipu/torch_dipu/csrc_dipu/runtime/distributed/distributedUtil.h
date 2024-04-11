// Copyright (c) 2024, DeepLink.
#pragma once

#include "csrc_dipu/runtime/core/DIPUGuard.h"

#include "DICLUtils.hpp"

namespace dipu {

class distributedUtil {
 public:
  virtual ~distributedUtil() = default;
  virtual void allreducePreFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                              std::vector<at::Tensor>& inputs,
                              std::vector<at::Tensor>& outputs){};

  virtual void allreducePostFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                               std::vector<at::Tensor>& inputs,
                               std::vector<at::Tensor>& outputs){};
};

}  // namespace dipu
