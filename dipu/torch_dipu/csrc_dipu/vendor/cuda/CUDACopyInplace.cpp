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

  void run(at::Tensor& dst, const at::Tensor& src, bool non_blocking) override {
    TORCH_CHECK(dst.defined(), "dst is undefined");
    TORCH_CHECK(src.defined(), "src is undefined");
    if (dst.numel() == 0 || dst.is_same(src)) {
      return;
    }
    auto info = CopyParamsInfo(dst, src);
    if (info.copyType_ == DIPUCopyType::D2Self) {
      non_blocking = true;
    }

// use synchronous copy on mx, see comment near RegisterDIPU.cpp 'BackendSelect'
#if DIPU_VENDOR_NAME_MUXI
    if (info.copyType_ == DIPUCopyType::D2H ||
        info.copyType_ == DIPUCopyType::H2D) {
      non_blocking = false;
    }
#endif

    // Exit early if dst and src are views of the same data
    if ((dst.is_alias_of(src) && dst.storage_offset() == src.storage_offset() &&
         info.sameStride_ && info.sameDtype_)) {
      return;
    }

    if (native::dumpOpArgLevel() > 1) {
      std::cout << "    DIPUCopyInplace.run:    dst:" << native::dumpArg(dst)
                << std::endl;
      std::cout << "    DIPUCopyInplace.run::   src:" << native::dumpArg(src)
                << std::endl;
    }

    switch (info.copyType_) {
      case DIPUCopyType::D2Self:
      case DIPUCopyType::D2OtherD:
        dipu_wrap_diopi_copy_inp(dst, src, non_blocking);
        break;
      default: {
        const DIPUGuard guard((!src.is_cpu()) ? src.device() : dst.device());
        auto curStream = dipu::getCurrentDIPUStream();
        info.updateCurrentStream(curStream);
        copyAll(dst, src, non_blocking, info);
        tryRecordOrSyncStream(info, dst, src, curStream, non_blocking,
                              /* block_cpu_d2d = */ false,
                              /* block_cpu_h2d = */ false);
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
