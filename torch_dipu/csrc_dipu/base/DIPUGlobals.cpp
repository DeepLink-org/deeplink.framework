#include "DIPUGlobals.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/runtime/core/DIPUEventPool.h"
#include <iostream>
namespace dipu {

void releaseAllResources() {
    releaseAllDeviceMem();
    releaseGlobalEventPool();
}

} // namespace dipu