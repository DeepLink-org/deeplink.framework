#include "DIPUGlobals.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/runtime/core/DIPUEventPool.h"
#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include <iostream>
#include <ctime>
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

} // namespace dipu