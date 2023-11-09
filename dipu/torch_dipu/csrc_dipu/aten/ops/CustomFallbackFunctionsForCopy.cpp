#include <ATen/ATen.h>

#include "csrc_dipu/aten/RegisterDIPU.hpp"

namespace dipu {
namespace native {
namespace {
  inline void try_record_stream(at::Tensor& self, DIPUStream& stream, bool is_default_stream) {
    if (self.is_cpu() && self.options().pinned_memory()) {
      self.record_stream(stream);
    } else if (!is_default_stream) {
      self.record_stream(stream);
    }
  }
}

static at::Tensor& custom_fallback_dipu_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {

  DIPU_OP_LOG_WARNING_ONCE("custom fallback to dipu copy, name=copy_" << std::endl);
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  static bool use_slow_copy = (std::getenv("DIPU_USE_SLOW_COPY") != nullptr);
  dipu::DIPUGuard guard(self.is_cpu() ? src.device() : self.device());
  if (non_blocking) {
    auto stream = dipu::getCurrentDIPUStream();
    const bool is_default_stream = dipu::getDefaultDIPUStream() == stream;
    try_record_stream(self, is_default_stream);
    try_record_stream(src, is_default_stream);
  }
  return dipu::getDipuCopyInplace()->run(self, src, non_blocking, use_slow_copy);
}

}  // namespace native
}  // namespace dipu
