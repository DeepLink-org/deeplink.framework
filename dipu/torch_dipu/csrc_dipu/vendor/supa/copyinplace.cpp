// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/aten/ops/DIPUCopy.hpp>
#include <csrc_dipu/common.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {
namespace supa {

using dipu::native::dipu_wrap_diopi_copy_inp;

// supa's existing implementaion same as cuda, it proxy all copy case to diopi,
// it's different with diopiCopy doc's requirement (only handle device copy),
// so we change it's behavior as only do device copy.
class SUPACopyInplace : public DIPUCopyInpOnDIOPI {
 public:
  SUPACopyInplace() = default;
  ~SUPACopyInplace() = default;

  // assume it can handle between device.
  void copyNodirectBetweenDevices(at::Tensor& dst, const at::Tensor& src,
                                  bool non_blocking,
                                  CopyParamsInfo& info) override {
    dipu_wrap_diopi_copy_inp(dst, src, non_blocking);
  }
  void run(at::Tensor& dst, const at::Tensor& src, bool non_blocking) override {
    auto curStream = dipu::getCurrentDIPUStream();
    ::diopiContext context(curStream.rawstream());
    auto ctx = &context;
    auto diopi_src = dipu::diopi_helper::toDiopiTensorHandle(src);
    auto diopi_dst = dipu::diopi_helper::toDiopiTensorHandle(dst);
    TORCH_CHECK(diopiError_t::diopiSuccess ==
                diopiCopyInp(ctx, diopi_src, diopi_dst));
    // syncAfterCopy
    if (!non_blocking) {
      dipu::devapis::syncStream(curStream.rawstream());
    }
  }
};

static SUPACopyInplace copy_inplace;
static int32_t supa_copy_inplace_init = []() {
  setDipuCopyInstance(&copy_inplace);
  return 1;
}();

}  // namespace supa
}  // namespace dipu
