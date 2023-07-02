// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "DIPUStream.h"
#include <map>

namespace dipu {

class DIPU_API CacheAllocator: public c10::Allocator {
  c10::Allocator* raw_allocator_ = nullptr;

  // maximum memory allowed to be cached
  const size_t limit_ ;

  // cached memory.
  std::atomic<size_t> currCached_ { 0 };
  size_t peakCached_ { 0 };

  // allocated memory.
  std::atomic<size_t> currAllocated_ { 0 };
  size_t peakAllocated_ { 0 };
  protected:


  protected:
    c10::Allocator* raw_allocator() const {
      return raw_allocator_;
    }

    void* allocate_raw(size_t n) {
      return raw_allocator()->raw_allocate(n);
    }

    void free_raw(void* ptr) {
      return raw_allocator()->raw_deallocate(ptr);
    }

  public:
    size_t cached() {
      return currCached_;
    }

    size_t available() {
      return limit_ - currCached_;
    }

    explicit CacheAllocator(c10::Allocator* raw_allocator, size_t limit = 1 << 30):raw_allocator_(raw_allocator), limit_(limit) {
      TORCH_CHECK(raw_allocator_);
    }

    virtual ~CacheAllocator() {};
};



}  // namespace dipu

