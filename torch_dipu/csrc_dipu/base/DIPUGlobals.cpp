#include "DIPUGlobals.h"

#include <ctime>
#include <iostream>

#include "csrc_dipu/runtime/core/DIPUEventPool.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
namespace dipu {

const char* getDipuCommitId() { return DIPU_GIT_HASH; }

static void printPromptAtStartup() {
    auto time = std::time(nullptr);
    std::string time_str = std::ctime(&time);
    std::cout << time_str.substr(0, time_str.size() - 1) << " dipu | git hash:" << getDipuCommitId() << std::endl;
}

void initResource() { printPromptAtStartup(); }

void releaseAllResources() {
    DIPU_DEBUG_ALLOCATOR(2, "releaseAllResources");
    releaseAllDeviceMem();
    releaseAllEvent();
}

}  // namespace dipu