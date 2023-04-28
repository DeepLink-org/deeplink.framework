#pragma once

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <chrono>

#include "dtu_utils.h"

namespace enflame {
builder::Op Conv2D_Grad(std::shared_ptr<builder::Builder> tmp_builder, builder::Op out_grad_, builder::Op input_, builder::Op filter_, 
                        std::vector<int64_t> bias_shape, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation);
}  // namespace enflame
