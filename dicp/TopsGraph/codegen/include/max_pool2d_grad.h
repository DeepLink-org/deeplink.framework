#pragma once

#include <memory>
#include <vector>
#include <string>

#include "common/dtu_utils.h"

namespace enflame {
builder::Op max_pool2d_grad(std::shared_ptr<builder::Builder> tmp_builder, builder::Op out_grad, builder::Op in,
                     std::vector<int> ksize, std::vector<int> strides, std::vector<int> padding);

builder::Op batch_norm(std::shared_ptr<builder::Builder> tmp_builder, builder::Op para1,
                       builder::Op para2, builder::Op para3);
}  // namespace enflame