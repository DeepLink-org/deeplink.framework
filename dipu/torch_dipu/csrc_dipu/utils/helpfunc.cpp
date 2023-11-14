// Copyright (c) 2023, DeepLink.
#include "./helpfunc.hpp"

#ifndef WIN32
#include <mutex>
#include <pthread.h>
#endif

namespace dipu {
bool isDeviceTensor(const at::Tensor &tensor) {
  // same as tensor.device().type()
  return tensor.unsafeGetTensorImpl()->device_type() == dipu::DIPU_DEVICE_TYPE;
}

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