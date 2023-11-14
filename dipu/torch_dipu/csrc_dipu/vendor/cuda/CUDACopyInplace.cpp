// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/common.h>
#include <c10/util/Exception.h>
#include <csrc_dipu/aten/ops/DIPUCopy.h>

namespace dipu {

using dipu::native::dipu_wrap_diopi_copy_inp;
class CUDACopyInplace: public DIPUCopyInplace<true, false> {
public:
  CUDACopyInplace() = default;
  ~CUDACopyInplace() = default;

  // diopi-cuda copy use aten, it handle all case. so leave all to it.
  void copyAll(at::Tensor& self, const at::Tensor& src,
              bool non_blocking, CopyParamsInfo& info) override {
      dipu_wrap_diopi_copy_inp(self, src, non_blocking);
  }
};

static CUDACopyInplace cuda_copy_inplace;
static int32_t cuda_init = []() {
  // setDipuCopyClass(&cuda_copy_inplace);
  return 1;
}();

}  // namespace dipu
