#include <stdio.h>

#include <torch/library.h>

#include "torch_dipu/csrc/aten/DIPUNativeFunctions.h"
#include "torch_dipu/csrc/aten/util/diopi.h"
#include "torch_dipu/csrc/aten/util/Log.h"

namespace at {

namespace {

namespace {

at::Tensor wrapperTensorAdd(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
    return dipu::native::DIPUNativeFunctions::add(self, other, alpha);
}

}  // inner anonymous namespace

TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    if (reinterpret_cast<void*>(diopiAdd) != nullptr) {
        m.impl("add.Tensor", TORCH_FN(wrapperTensorAdd));
    } else {
        DIPU_LOG << "diopiAdd not implemented, do not register\n";
    }
}

}  // outer anonymous namespace

}  // namespace at