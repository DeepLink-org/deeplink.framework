// Copyright (c) 2023, DeepLink.

#include "DIPUCopy.hpp"

#include <algorithm>

#include <c10/util/Exception.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>

namespace dipu {

static DIPUCopyInplace<true, false> default_copy_inplace_op;
static DIPUCopyBase* dipu_copy_op = &default_copy_inplace_op;

DIPUCopyBase* getDipuCopyInstance() {
  TORCH_CHECK(dipu_copy_op, "dipu copy inplace not registered");
  return dipu_copy_op;
}

void setDipuCopyInstance(DIPUCopyBase* op) { dipu_copy_op = op; }

}  // namespace dipu

namespace dipu::native {
at::Scalar DIPUATenFunctions::_local_scalar_dense_dipu(const at::Tensor& self) {
  at::Scalar r;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::kHalf, at::kBool, self.scalar_type(), "_local_scalar_dense_dipu",
      [&] {
        scalar_t value;
        dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
        MemChecker::instance().check(self);
        dipu::devproxy::memCopyD2HAsync(stream.rawstream(), sizeof(scalar_t),
                                        &value, self.data_ptr<scalar_t>());
        dipu::devproxy::syncStream(stream.rawstream());
        r = at::Scalar(value);
      });
  return r;
}
}  // namespace dipu::native
