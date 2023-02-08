#include "allocator.h"
#include "DIPUCachingAllocator.h"

namespace torch_dipu {

static DIPUAllocator allocator;

c10::Allocator* getDIPUAllocator(void) {
  return &allocator;
}

// using in at::empty
static DIPUCachingAllocator cache_allocator;

c10::Allocator* getDIPUCachingAllocator(void) {
  return &cache_allocator;
}

REGISTER_ALLOCATOR(torch_dipu::DIPU_DEVICE_TYPE, &allocator);

}  // namespace torch_dipu