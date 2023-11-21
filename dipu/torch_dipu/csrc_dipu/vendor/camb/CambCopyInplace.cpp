// Copyright (c) 2023, DeepLink.

#include <c10/util/Exception.h>

#include <csrc_dipu/aten/ops/DIPUCopy.hpp>
#include <csrc_dipu/common.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {

using dipu::native::dipu_wrap_diopi_copy_inp;
class CambCopyInplace : public DIPUCopyInplace<true, false> {
 public:
  CambCopyInplace() = default;
  ~CambCopyInplace() = default;

  // diopicamb copy has restriction ^#^
  void copyNodirectOnDevice(at::Tensor& dst, const at::Tensor& src,
                            bool non_blocking, CopyParamsInfo& info) override {
    if ((dst.is_complex() || src.is_complex())) {
      doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
    } else {
      DIPUCopyInplace::copyNodirectOnDevice(dst, src, non_blocking, info);
    }
  }
};

static CambCopyInplace camb_copy_inplace;
static int32_t cuda_init = []() {
  setDipuCopyClass(&camb_copy_inplace);
  return 1;
}();

}  // namespace dipu
