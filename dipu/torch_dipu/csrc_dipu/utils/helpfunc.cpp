// Copyright (c) 2023, DeepLink.
#include "./helpfunc.hpp"

#include <c10/core/TensorImpl.h>

#include <Python.h>

#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/runtime/core/DIPUStorageImpl.h"

#ifndef WIN32
#include <mutex>
#include <pthread.h>
#endif

namespace dipu {
bool isDeviceTensor(const at::Tensor& tensor) {
  // same as tensor.device().type()
  return tensor.unsafeGetTensorImpl()->device_type() == dipu::DIPU_DEVICE_TYPE;
}

diopiMemoryFormat_t get_format(const at::Tensor& tensor) {
  if (!isDeviceTensor(tensor)) {
    return diopiMemoryFormat_t::Undefined;
  }
  ::diopiStorageDesc_t desc;
  dipu::DIPUStorageImpl::GetImplPtr(reinterpret_cast<const at::Tensor&>(tensor))
      ->get_desc(&desc);
  return desc.format;
}

at::Tensor format_cast(at::Tensor tensor, diopiMemoryFormat_t target_format) {
  TORCH_CHECK(isDeviceTensor(tensor), "only device tensor support this api.");
  TORCH_CHECK(::diopiFormatCast, "diopi not support this api.");
  ::diopiTensorHandle_t input = dipu::diopi_helper::toDiopiTensorHandle(tensor);
  ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
  ::diopiTensorHandle_t out = nullptr;
  if (PyGILState_Check() != 0) {
    Py_BEGIN_ALLOW_THREADS ::diopiFormatCast(&context, &out, input,
                                             target_format);
    Py_END_ALLOW_THREADS
  }
  return *(reinterpret_cast<at::Tensor*>(out));
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
