// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "DIPUStream.h"

namespace dipu {

// A caching allocator for host allocations (pinned memory).
//
// The allocator re-uses freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to dipu::devapis::freeHost calls.
//
// To ensure correct behavior, CachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a MemcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by dipu::native::DIPUATenFunctions::copy_.
//
// Note that this allocator does not split larger allocations into smaller
// blocks, unlike the caching device allocator.
DIPU_API c10::Allocator* getCachingHostAllocator();

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
DIPU_API bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, DIPUStream stream);

// Releases cached pinned memory allocations via dipu::devapis::freeHost
DIPU_API void CachingHostAllocator_emptyCache();

inline DIPU_API at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}

inline DIPU_API c10::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}

}  // namespace dipu
