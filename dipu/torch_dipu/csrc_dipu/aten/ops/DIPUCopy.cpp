// Copyright (c) 2023, DeepLink.

#include "DIPUCopy.hpp"

#include <algorithm>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"

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
#if DIPU_VENDOR_NAME_SUPA
  extern void setSupaDeviceCopyInfo(void* ptr, int64_t offset);
  if (self.scalar_type() == c10::ScalarType::Bool) {
    // on SUPA, bool type is represents by float.
    using scalar_t = c10::impl::ScalarTypeToCPPTypeT<at::kFloat>;
    float value;
    dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
    MemChecker::instance().check(self);
    /* on SUPA, it can't plus offset to ptr since it is virtual address.
       must set offset in advance and recaculate ptr after translating it to
       physical address.
    */
    setSupaDeviceCopyInfo(self.storage().data(),
                          self.storage_offset() * sizeof(scalar_t));
    dipu::devproxy::memCopyD2HAsync(stream.rawstream(), sizeof(scalar_t),
                                    &value, self.data_ptr());
    dipu::devproxy::syncStream(stream.rawstream());
    r = at::Scalar((std::abs(value) >= 1e-6f));
    return r;
  }
#endif
  AT_DISPATCH_ALL_TYPES_AND3(
      at::kHalf, at::kBool, at::kBFloat16, self.scalar_type(),
      "_local_scalar_dense_dipu", [&] {
        scalar_t value;
        dipu::DIPUStream stream = dipu::getCurrentDIPUStream();
        MemChecker::instance().check(self);
#if DIPU_VENDOR_NAME_ASCEND
        dipu::devproxy::syncStream(stream.rawstream());
        dipu::devproxy::memCopyD2H(sizeof(scalar_t), &value,
                                   self.data_ptr<scalar_t>());
#else
#if DIPU_VENDOR_NAME_SUPA
        setSupaDeviceCopyInfo(self.storage().data(), self.storage_offset()*sizeof(scalar_t));
#endif
        dipu::devproxy::memCopyD2HAsync(stream.rawstream(), sizeof(scalar_t),
                                        &value, self.data_ptr<scalar_t>());
        dipu::devproxy::syncStream(stream.rawstream());
#endif
        r = at::Scalar(value);
      });
  return r;
}
}  // namespace dipu_aten
}  // namespace native
}  // namespace dipu
