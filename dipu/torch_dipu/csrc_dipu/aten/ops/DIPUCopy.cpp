// Copyright (c) 2023, DeepLink.

#include "DIPUCopy.hpp"

#include <algorithm>

#include <c10/util/Exception.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>

namespace dipu {

// it's the default strategy and be assigned to the pointer dipu_copy_op_ which
// is a mutable poiner and may be change by vendor at runtime, so nor can
// this variable be const.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static DIPUCopyInpOnDIOPI default_copy_inplace_op;

auto& dipu_copy_op() {
  // the default strategy can be changed by vendor, so dipu_copy_op_ is not
  // const but a mutable poiner.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static DIPUCopyBase* dipu_copy_op_ = &default_copy_inplace_op;
  return dipu_copy_op_;
}

DIPUCopyBase* getDipuCopyInstance() {
  TORCH_CHECK(dipu_copy_op(), "dipu copy inplace not registered");
  return dipu_copy_op();
}

void setDipuCopyInstance(DIPUCopyBase* op) { dipu_copy_op() = op; }

}  // namespace dipu

namespace dipu {
namespace native {
namespace dipu_aten {
at::Scalar _local_scalar_dense_dipu(const at::Tensor& self) {
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
}  // namespace dipu_aten
}  // namespace native
}  // namespace dipu
