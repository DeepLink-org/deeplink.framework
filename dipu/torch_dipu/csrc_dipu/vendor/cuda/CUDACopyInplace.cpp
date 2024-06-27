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
  void copyPostProcess(const at::Tensor& dst, const at::Tensor& src,
                       bool non_blocking, const CopyParamsInfo& info,
                       DIPUStream& curStream) override {
    // 1. block_cpu_d2d=False on cuda, because we do not need sync stream when
    // copy on two devices, just wait between stream
    // 2. block_cpu_h2d=False on cuda, We do not need sync stream if cpu tensor
    // is not pin memory which stay consistent with
    // aten/src/ATen/native/cuda/Copy.cu.
    tryRecordOrSyncStream(info, dst, src, curStream, non_blocking,
                          /* block_cpu_d2d = */ false,
                          /* block_cpu_h2d = */ false);
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
