#pragma once

#include <memory>
#include <vector>
#include <string>

#include "dtu_utils.h"

namespace enflame {
builder::Op MaxPool2D(
    std::shared_ptr<builder::Builder> tmp_builder,
    builder::Op input,
    std::vector<long> ksize,
    std::vector<long> strides,
    std::vector<long> padding,
    std::vector<long> shape);

builder::Op MaxPool2D_Grad(
    std::shared_ptr<builder::Builder> tmp_builder,
    builder::Op out_grad,
    builder::Op in,
    std::vector<int> ksize,
    std::vector<int> strides,
    std::vector<int> padding);
}  // namespace enflame