#include "DIPUGlobals.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/runtime/core/DIPUEventPool.h"
#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include <iostream>
#include <ctime>
#ifndef WIN32
#include <pthread.h>
#include <mutex>
#endif

namespace dipu {

const char* getDipuCommitId() {
  return DIPU_GIT_HASH;
}

static void printPromptAtStartup() {
  auto time = std::time(nullptr);
  std::string time_str = std::ctime(&time);
  std::cout << time_str.substr(0, time_str.size() - 1) << " dipu | git hash:" << getDipuCommitId() << std::endl;
}

void initResource() {
  printPromptAtStartup();
  devproxy::initializeVendor();
  initCachedAllocator();
  at::DIPUOpRegister::register_op();
}

void releaseAllResources() {
  releaseAllDeviceMem();
  releaseAllEvent();
  devproxy::finalizeVendor();
}

static bool in_bad_fork = false;

#include <iostream>
bool is_in_bad_fork() {
  std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  return in_bad_fork;
}

#ifndef WIN32
// Called in the forked child if device has already been initialized
static void forked_child() {
  std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  in_bad_fork = true;
}
#endif

// Should be called before the first device call.
// Note: This is distinct from initExtension because a stub device implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
void poison_fork() {
  std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
#ifndef WIN32
  static std::once_flag flag;
  std::call_once(flag, []{ pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

} // namespace dipu