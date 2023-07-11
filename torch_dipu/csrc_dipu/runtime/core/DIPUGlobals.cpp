#include "DIPUGlobals.h"
#include "DIPUCachingAllocator.h"
#include "DIPUEventPool.h"
#include <iostream>
namespace dipu {

void releaseAllResources() {
    std::cout << __FUNCTION__ << std::endl;
    releaseAllDeviceMem();
}

} // namespace dipu