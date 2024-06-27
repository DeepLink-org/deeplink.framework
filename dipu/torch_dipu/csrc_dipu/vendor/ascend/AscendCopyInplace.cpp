// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/aten/ops/DIPUCopy.hpp>
#include <csrc_dipu/runtime/core/DIPUGuard.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {

class AscendCopyInplace : public DIPUCopyInpOnDIOPI {
 public:
  AscendCopyInplace() = default;
  ~AscendCopyInplace() override = default;

 protected:
  void copyPreProcess(const at::Tensor& dst, const at::Tensor& src,
                      bool non_blocking, CopyParamsInfo& info) override {
    if (!non_blocking && (DIPUCopyType::H2D == info.copyType_ ||
                          DIPUCopyType::D2H == info.copyType_)) {
      // According to our benchmark for H2D/D2H synchronous direct memory copy,
      // (Sync + memCopySync) is faster than (memCopyAsync + Sync) on Ascend,
      // So do an advance sync here
      info.curStream_.synchronize();
    }
    if (DIPUCopyType::D2OtherD == info.copyType_) {
      doSrcStreamWaitDstStream(info, true);
    }
  }

  void directMemCopy(at::Tensor& dst, const at::Tensor& src,
                     CopyParamsInfo& info, bool non_blocking) override {
    if (!non_blocking && (DIPUCopyType::H2D == info.copyType_ ||
                          DIPUCopyType::D2H == info.copyType_)) {
      // According to our benchmark for H2D/D2H synchronous direct memory copy,
      // (Sync + memCopySync) is faster than (memCopyAsync + Sync) on Ascend,
      // so do a memCopySync instead of memCopyAsync here
      memCopy(dst, src, info.curStream_, info.copyType_,
              /*nonOverlappingAndDense=*/true, /*isSynchronousCopy=*/true);
    } else {
      doDirectMemCopy(dst, src, info.curStream_, info.copyType_,
                      /*needMemCpSync=*/false);
    }
  }

  void copyPostProcess(const at::Tensor& dst, const at::Tensor& src,
                       bool non_blocking, const CopyParamsInfo& info,
                       DIPUStream& curStream) override {
    // In d2self cases, non_blocking has no effect (Ref:
    // https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html). In
    // d2h/h2d cases, the (Sync + memCopySync) strategy is adopted (see the
    // comments in the above functions copyPreProcess and directMemCopy), so
    // synchronization is never needed here.
    // record after copy
    if (non_blocking) {
      tryRecordOrSyncStream(dst, src, curStream, true);
    }

    if (DIPUCopyType::D2OtherD == info.copyType_) {
      doDstStreamWaitSrcStream(info, true);
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
