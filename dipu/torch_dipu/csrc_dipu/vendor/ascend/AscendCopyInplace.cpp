// Copyright (c) 2023, DeepLink.

#include <c10/util/Exception.h>

#include <csrc_dipu/aten/ops/DIPUCopy.hpp>
#include <csrc_dipu/common.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {

using dipu::native::dipu_wrap_diopi_copy_inp;
class AscendCopyInplace : public  DIPUCopyInpOnDIOPIWithCast{
 public:
  AscendCopyInplace() = default;
  ~AscendCopyInplace() override = default;

  // diopi-ascend copy use aten, so it can handle between-device case.
  void copyNodirectBetweenDevices(at::Tensor& dst, const at::Tensor& src,
                                  bool non_blocking,
                                  CopyParamsInfo& info) override {
    dipu_wrap_diopi_copy_inp(dst, src, non_blocking);
  }

 protected:
  void copyPostProcess(bool non_blocking, const CopyParamsInfo& info,
                       DIPUStream& curStream) override {
  }

  // overriding this func is possible but not recommended
  virtual void copyAll(at::Tensor& dst, const at::Tensor& src,
                       bool non_blocking, CopyParamsInfo& info) {
      dipu_wrap_diopi_copy_inp(dst, src, non_blocking);
  }
};

// not const, see comments in DIPUCopy.cpp dipu_copy_op()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static AscendCopyInplace ascend_copy_inplace;

// this variable only for call setInst. no other use
const static int32_t ascend_init = []() {
  setDipuCopyInstance(&ascend_copy_inplace);
  return 1;
}();

}  // namespace dipu
