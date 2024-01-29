// Copyright (c) 2023, DeepLink.

#include <c10/util/Exception.h>

#include <csrc_dipu/aten/ops/DIPUCopy.hpp>
#include <csrc_dipu/common.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {

using dipu::native::dipu_wrap_diopi_copy_inp;
class CUDACopyInplace : public DIPUCopyInpOnDIOPI {
 public:
  CUDACopyInplace() = default;
  ~CUDACopyInplace() override = default;

  // diopi-cuda copy use aten, so it can handle between-device case.
  void copyNodirectBetweenDevices(at::Tensor& dst, const at::Tensor& src,
                                  bool non_blocking,
                                  CopyParamsInfo& info) override {
    dipu_wrap_diopi_copy_inp(dst, src, non_blocking);
  }

 protected:
  void copyPostProcess(bool non_blocking, const CopyParamsInfo& info,
                       DIPUStream& curStream) override {
    // syncAfterCopy
    if (!non_blocking) {
      // for d2d cases, ignoring `non_blocking` for better performance
      if (info.copyType_ != DIPUCopyType::D2Self) {
        dipu::devapis::syncStream(curStream.rawstream());
      }
    }
  }
};

// not const, see comments in DIPUCopy.cpp dipu_copy_op()
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static CUDACopyInplace cuda_copy_inplace;

// this variable only for call setInst. no other use
const static int32_t cuda_init = []() {
  setDipuCopyInstance(&cuda_copy_inplace);
  return 1;
}();

}  // namespace dipu
