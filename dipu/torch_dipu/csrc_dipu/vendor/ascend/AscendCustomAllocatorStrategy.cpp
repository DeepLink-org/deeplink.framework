#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"

const static int ascend_init = []() {
  // – 内存大小向上对齐成32整数倍+32字节(m=ALIGN_UP[len,32]+32字节);
  // – 内存起始地址需满足64字节对齐(ALIGN_UP[m,64])。
  static dipu::MemoryAlignmentStrategy memoryAlignStrategy;
  //  nbytes = alpha * nbytes + beta;
  memoryAlignStrategy.kBytesAlign = 64;
  memoryAlignStrategy.beta = 32;
  memoryAlignStrategy.alpha = 1;
  dipu::setMemoryAlignmentStrategy(&memoryAlignStrategy);
  return 0;
}();
