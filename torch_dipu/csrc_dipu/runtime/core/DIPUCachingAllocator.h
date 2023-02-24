#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <caffe2/core/logging.h>

#include "DIPUStream.h"

namespace dipu {

class DIPUCachingAllocator : public c10::Allocator {
public:
  c10::DataPtr allocate(size_t size) const override;
  c10::DeleterFnPtr raw_deleter() const override;
protected:
  c10::DataPtr allocate(size_t size, c10::DeviceIndex device_id) const;
};

uint64_t currentMemoryAllocated(int device_id);
uint64_t currentMemoryCached(int device_id);
uint64_t maxMemoryAllocated(int device_id);
uint64_t maxMemoryCached(int devcurrentMemoryAllocatedice_id);
void emptyCachedMem();
void setDebugEnv(char* flag);
void memoryDebug(c10::DataPtr* data);
void memoryDebug(const c10::DataPtr* data);
void memoryDebug();
void recordStream(const c10::DataPtr& ptr, dipu::DIPUStream stream);
bool get_memory_strategy();
void set_memory_strategy(bool ms);

}  // namespace dipu
