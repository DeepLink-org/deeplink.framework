// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/aten/ops/DIPUCopy.hpp>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {

class AscendCopyInplace : public DIPUCopyInpOnDIOPI {
 public:
  AscendCopyInplace() = default;
  ~AscendCopyInplace() override = default;

 protected:
  void copyPostProcess(bool non_blocking, const CopyParamsInfo& info,
                       DIPUStream& curStream) override {
    // Ref: https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html
    // In d2self cases, non_blocking has no effect.
    // For other cases, do sync after copy if non_blocking is false.
    if (!non_blocking && info.copyType_ != DIPUCopyType::D2Self) {
      dipu::devapis::syncStream(curStream.rawstream());
    }
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
