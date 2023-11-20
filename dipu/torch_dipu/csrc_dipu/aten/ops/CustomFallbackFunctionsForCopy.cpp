#include <ATen/ATen.h>

#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include "csrc_dipu/aten/ops/DIPUCopy.h"

namespace dipu {
namespace native {

static DIPUCopyInplace<false, false> onCpuCopy;
static at::Tensor& custom_fallback_dipu_copy_(at::Tensor& self,
                                              const at::Tensor& src,
                                              bool non_blocking) {
  DIPU_OP_LOG_WARNING_ONCE("custom fallback to dipu copy, name=copy_"
                           << std::endl);
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  dipu::DIPUGuard guard(self.is_cpu() ? src.device() : self.device());
  onCpuCopy.run(self, src, non_blocking);
  return self;
}

}  // namespace native
}  // namespace dipu
