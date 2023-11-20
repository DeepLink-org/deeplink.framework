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
class SUPACopyInplace : public DIPUCopyInplace<true, false> {
 public:
  SUPACopyInplace() = default;
  ~SUPACopyInplace() = default;

  void copyNodirectOnDevice(at::Tensor& dst, const at::Tensor& src,
                            bool non_blocking, CopyParamsInfo& info) override {
    dipu_wrap_diopi_copy_inp(dst, src, non_blocking);
  }
};

static SUPACopyInplace copy_inplace;
static int32_t suap_copy_inplace_init = []() {
  setDipuCopyClass(&copy_inplace);
  return 1;
}();

}  // namespace supa
}  // namespace dipu
