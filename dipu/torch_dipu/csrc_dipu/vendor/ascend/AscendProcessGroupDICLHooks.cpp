// Copyright (c) 2024, DeepLink.

#include <c10/core/ScalarType.h>

#include "csrc_dipu/runtime/distributed/ProcessGroupDICL.h"

namespace dipu {

namespace dicl_hook {

void allReducePreFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                    std::vector<at::Tensor>& inputs,
                    std::vector<at::Tensor>& outputs) {
  if (inputs[0].scalar_type() == at::kByte) {
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0] = inputs[0].to(at::kShort);
  } else if ((inputs[0].scalar_type() == at::kBool)) {
    // To avoid overflow when summing bool data types, change the value of True
    // to 1 and change the value of other places to 0. Generally, world_size
    // will not exceed 256, so there will be no overflow.
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0].ne_(0);
  }
}

void allReducePostFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                     std::vector<at::Tensor>& inputs,
                     std::vector<at::Tensor>& outputs) {
  if (inputs[0].scalar_type() != outputs[0].scalar_type()) {
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0].copy_(inputs[0]);
  } else if ((inputs[0].scalar_type() == at::kBool)) {
    // Make the bool type behavior aligned with cuda (1 is True, 0 is False)
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0].ne_(0);
  }
}

void reducePreFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                 std::vector<at::Tensor>& inputs,
                 std::vector<at::Tensor>& outputs) {
  if (inputs[0].scalar_type() == at::kByte) {
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0] = inputs[0].to(at::kShort);
  }
}

void reducePostFn(std::vector<std::shared_ptr<DICLComm>>& comms,
                  std::vector<at::Tensor>& inputs,
                  std::vector<at::Tensor>& outputs) {
  if (inputs[0].scalar_type() != outputs[0].scalar_type()) {
    DIPUStreamGuard guard(comms[0]->diclStream_.unwrap());
    outputs[0].copy_(inputs[0]);
  }
}

}  // namespace dicl_hook

}  // namespace dipu
