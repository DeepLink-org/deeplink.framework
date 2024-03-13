#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"

const static int n = []() {
    // – 内存大小向上对齐成32整数倍+32字节(m=ALIGN_UP[len,32]+32字节);
    // – 内存起始地址需满足64字节对齐(ALIGN_UP[m,64])。
    static dipu::MemoryAlignmentStrategy allocateParam;
    //  nbytes = alpha * nbytes + beta;
    allocateParam.kBytesAlign = 64;
    allocateParam.beta = 32;
    dipu::setMemoryAlignmentStrategy(&allocateParam);
    return 0;
}();