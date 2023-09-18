#include "DIPUGlobals.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/runtime/core/DIPUEventPool.h"
#include "csrc_dipu/runtime/core/DIPUGeneratorImpl.h"
#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include <iostream>
#include <ctime>
#include <atomic>
namespace dipu {

const char* getDipuCommitId() {
  return DIPU_GIT_HASH;
}

static void printPromptAtStartup() {
  auto time = std::time(nullptr);
  std::string time_str = std::ctime(&time);
  std::cout << time_str.substr(0, time_str.size() - 1) << " dipu | git hash:" << getDipuCommitId() << std::endl;
}

static void initResourceImpl() {
  static std::atomic_bool called(false);
  if (called == true) {
    return;
  }
  called = true;

  printPromptAtStartup();
  devproxy::initializeVendor();
  initCachedAllocator();
  at::DIPUOpRegister::register_op();
}

static void releaseAllResourcesImpl() {
  static std::atomic_bool called(false);
  if (called == true) {
    return;
  }
  called = true;
  releaseAllGenerator();
  releaseAllDeviceMem();
  releaseAllEvent();
  devproxy::finalizeVendor();
}

namespace {
  class DIPUIniter {
  public:
    DIPUIniter() {
      initResourceImpl();
    }

    ~DIPUIniter() {
      releaseAllResourcesImpl();
    }
  };
} // namespace

void initResource() {
 initResourceImpl();
 /* In some cases(eg: spawn process), the resource cleanup function we registered will not be executed,
    so we use the destructor of the static variable in the function here just in case. */
 static DIPUIniter initer;
}

void releaseAllResources() {
 releaseAllResourcesImpl();
}

} // namespace dipu