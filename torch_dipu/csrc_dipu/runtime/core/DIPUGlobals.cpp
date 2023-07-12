#include "DIPUGlobals.h"
#include "DIPUCachingAllocator.h"
#include "DIPUEventPool.h"
#include <iostream>
namespace dipu {

void releaseAllResources() {
    releaseAllDeviceMem();
    releaseGlobalEventPool();
}

} // namespace dipu