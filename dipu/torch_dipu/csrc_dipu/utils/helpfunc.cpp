// Copyright (c) 2023, DeepLink.
#include "helpfunc.hpp"

#include <cstdint>
#include <mutex>
#include <pthread.h>

#include <ATen/core/TensorBody.h>
#include <c10/util/Exception.h>

#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_ext.h>

#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/runtime/core/DIPUStream.h"
#include "csrc_dipu/vendor/vendorapi.h"  // IWYU pragma: keep

namespace dipu {
bool isDeviceTensor(const at::Tensor& tensor) {
  // same as tensor.device().type()
  return tensor.unsafeGetTensorImpl()->device_type() == dipu::DIPU_DEVICE_TYPE;
}

at::Tensor native_memory_format_cast(at::Tensor tensor,
                                     NativeMemoryFormat_t format) {
  TORCH_CHECK(isDeviceTensor(tensor), "only device tensor support this api.");
  TORCH_CHECK(::diopiNativeMemoryFormatCast, "diopi not support this api.");
  ::diopiTensorHandle_t in = diopi_helper::toDiopiTensorHandle(tensor);
  ::diopiContext context(getCurrentDIPUStream().rawstream());
  ::diopiTensorHandle_t out = nullptr;
  ::diopiNativeMemoryFormatCast(&context, &out, in,
                                static_cast<int64_t>(format));
  return *(diopi_helper::fromDiopiTensorHandle(out));
}

NativeMemoryFormat_t get_native_memory_format(const at::Tensor& tensor) {
  TORCH_CHECK(::diopiGetNativeMemoryFormat, "diopi not support this api.");
  ::diopiContext context(getCurrentDIPUStream().rawstream());
  ::diopiConstTensorHandle_t input = diopi_helper::toDiopiTensorHandle(tensor);
  int64_t format = -1;
  ::diopiGetNativeMemoryFormat(&context, input, &format);
  return static_cast<NativeMemoryFormat_t>(format);
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
