#include "DIPUGlobals.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include <iostream>
namespace dipu {

void releaseAllResources() {
    releaseAllDeviceMem();
}

} // namespace dipu