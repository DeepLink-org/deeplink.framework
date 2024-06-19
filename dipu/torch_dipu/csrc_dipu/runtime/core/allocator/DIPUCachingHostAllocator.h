// Copyright (c) 2024, DeepLink.
#pragma once

#include <c10/core/Allocator.h>

#include "csrc_dipu/runtime/core/DIPUStream.h"

namespace dipu::allocator {

// ----------------------------------------------------------------------------
// Code from pytorch2.1.0 aten/src/ATen/cuda/CachingHostAllocator.h
// ----------------------------------------------------------------------------

//
// A caching allocator for host allocations (pinned memory).
//
// This reuses freed pinned (page-locked) memory allocations.
// This avoids device synchronizations due to freeHost calls.
//
// To ensure correct behavior, CachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a memcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by copy_.
//
// Note that this allocator does not split larger allocations into smaller
// blocks, unlike the caching device allocator.
//
c10::Allocator* getCachingHostAllocator();

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
bool CachingHostAllocator_recordEvent(void* ptr, void* ctx,
                                      const DIPUStream& stream);

// Releases cached pinned memory allocations via freeHost
void CachingHostAllocator_emptyCache();

bool CachingHostAllocator_isPinnedPtr(const void* ptr);

}  // namespace dipu::allocator
