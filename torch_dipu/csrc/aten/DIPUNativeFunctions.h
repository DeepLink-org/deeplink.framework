# pragma once

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace dipu::native {

struct DIPUNativeFunctions {
    static at::Tensor add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
};

}  // namespace dipu::native