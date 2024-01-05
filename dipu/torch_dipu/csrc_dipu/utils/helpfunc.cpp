// Copyright (c) 2023, DeepLink.
#include "./helpfunc.hpp"

#include <diopi/functions_ext.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#ifndef WIN32
#include <mutex>
#include <pthread.h>
#endif

namespace dipu {
bool isDeviceTensor(const at::Tensor& tensor) {
  // same as tensor.device().type()
  return tensor.unsafeGetTensorImpl()->device_type() == dipu::DIPU_DEVICE_TYPE;
}

at::Tensor format_cast(at::Tensor tensor, CustomFormat_t format) {
  TORCH_CHECK(isDeviceTensor(tensor), "only device tensor support this api.");
  TORCH_CHECK(::diopiCustomFormatCast, "diopi not support this api.");
  ::diopiTensorHandle_t in = diopi_helper::toDiopiTensorHandle(tensor);
  ::diopiContext context(getCurrentDIPUStream().rawstream());
  ::diopiTensorHandle_t out = nullptr;
  ::diopiCustomFormatCast(&context, &out, in, (int64_t)format);
  return *(reinterpret_cast<const at::Tensor*>(out));
}

CustomFormat_t get_format(const at::Tensor& tensor) {
  TORCH_CHECK(::diopiGetCustomFormat, "diopi not support this api.");
  ::diopiContext context(getCurrentDIPUStream().rawstream());
  ::diopiConstTensorHandle_t input = diopi_helper::toDiopiTensorHandle(tensor);
  int64_t format;
  ::diopiGetCustomFormat(&context, input, &format);
  return (CustomFormat_t)format;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static bool in_bad_fork = false;
bool is_in_bad_fork() { return in_bad_fork; }

#ifndef WIN32
// Called in the forked child if device has already been initialized
static void forked_child() { in_bad_fork = true; }
#endif

// Should be called before the first device call.
// Note: This is distinct from initExtension because a stub device
// implementation has some working functions (e.g. device_count) but cannot
// fully initialize.
void poison_fork() {
#ifndef WIN32
  static std::once_flag flag;
  std::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

}  // namespace dipu
