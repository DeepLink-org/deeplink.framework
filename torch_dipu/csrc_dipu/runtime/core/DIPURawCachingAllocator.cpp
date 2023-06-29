// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"

namespace dipu {

class RawCachingAllocator: public CacheAllocator {
public:
  RawCachingAllocator(c10::Allocator* raw_allocator): CacheAllocator(raw_allocator) {

  }

  ~RawCachingAllocator() {

  }
};


}  // namespace dipu