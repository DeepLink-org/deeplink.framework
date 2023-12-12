// Copyright (c) 2023, DeepLink.
#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include "csrc_dipu/runtime/core/DIPUStream.h"

namespace dipu {
size_t memoryReserved(const c10::Device& device);

size_t memoryAllocated(const c10::Device& device);

size_t maxMemoryReserved(const c10::Device& device);

size_t maxMemoryAllocated(const c10::Device& device);

void emptyCachedMem();

void initCachedAllocator();

void releaseAllDeviceMem();

void recordStream(const c10::DataPtr& ptr, const DIPUStream& stream);

void recordStream(const at::Tensor& tensor, const DIPUStream& stream);

}  // namespace dipu
