// Copyright (c) 2024, DeepLink.

#include "csrc_dipu/runtime/distributed/ProcessGroupDICL.h"

namespace dipu {

namespace dicl_hook {

void allReducePreFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                    std::vector<at::Tensor>& inputs,
                    std::vector<at::Tensor>& outputs) {
  if (inputs[0].scalar_type() == at::kBool ||
      inputs[0].scalar_type() == at::kByte) {
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0] = inputs[0].to(at::kInt);
  }
}

void allReducePostFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                     std::vector<at::Tensor>& inputs,
                     std::vector<at::Tensor>& outputs) {
  if (inputs[0].scalar_type() != outputs[0].scalar_type()) {
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0].copy_(inputs[0]);
  }
}

}  // namespace dicl_hook

}  // namespace dipu
