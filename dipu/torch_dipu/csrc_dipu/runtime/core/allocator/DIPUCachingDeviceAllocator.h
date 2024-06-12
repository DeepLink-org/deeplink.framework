// Copyright (c) 2024, DeepLink.

#pragma once

#include <array>
#include <iostream>
#include <mutex>
#include <set>
#include <unordered_set>
#include <utility>

#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/Registry.h>

#include "csrc_dipu/runtime/core/DIPUStream.h"

namespace dipu::allocator {

#if DIPU_TORCH_VERSION >= 20100
using GatheredContext = c10::GatheredContext;
#else
struct GatheredContext {
  virtual ~GatheredContext() = default;
};
#endif

// Caching allocator will execute every registered callback if it unable to find
// block inside of already allocated area.
class FreeMemoryCallback {
 public:
  virtual ~FreeMemoryCallback() = default;
  virtual bool Execute() = 0;
};

C10_DECLARE_REGISTRY(FreeDeviceMemoryCallbacksRegistry, FreeMemoryCallback);
#define REGISTER_FREE_MEMORY_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(FreeDeviceMemoryCallbacksRegistry, name, __VA_ARGS__);

struct Stat {
  int64_t current = 0;
  int64_t peak = 0;
  int64_t allocated = 0;
  int64_t freed = 0;
};

enum struct StatType : uint8_t {
  AGGREGATE = 0,
  SMALL_POOL = 1,
  LARGE_POOL = 2,
  NUM_TYPES = 3  // remember to update this whenever a new stat type is added
};

using StatArray = std::array<Stat, static_cast<size_t>(StatType::NUM_TYPES)>;

// Struct containing memory allocator summary statistics for a device.
struct DeviceStats {
  // COUNT: allocations requested by client code
  StatArray allocation;
  // COUNT: number of allocated segments from mallocDevice().
  StatArray segment;
  // COUNT: number of active memory blocks (allocated or used by stream)
  StatArray active;
  // COUNT: number of inactive, split memory blocks (unallocated but can't be
  // released via freeDevice)
  StatArray inactive_split;

  // SUM: bytes allocated by this memory alocator
  StatArray allocated_bytes;
  // SUM: bytes reserved by this memory allocator (both free and used)
  StatArray reserved_bytes;
  // SUM: bytes within active memory blocks
  StatArray active_bytes;
  // SUM: bytes within inactive, split memory blocks
  StatArray inactive_split_bytes;
  // SUM: bytes requested by client code
  StatArray requested_bytes;

  // COUNT: total number of failed calls to mallocDevice necessitating cache
  // flushes.
  int64_t num_alloc_retries = 0;

  // COUNT: total number of OOMs (i.e. failed calls to DEVICE after cache flush)
  int64_t num_ooms = 0;

  // COUNT: total number of oversize blocks allocated from pool
  Stat oversize_allocations;

  // COUNT: total number of oversize blocks requiring malloc
  Stat oversize_segments;

  // SIZE: maximum block size that is allowed to be split.
  int64_t max_split_size = 0;
};

using CreateContextFn = std::shared_ptr<GatheredContext> (*)();

// Struct containing info of an allocation block (i.e. a fractional part of a
// mallocDevice)..
struct BlockInfo {
  int64_t size = 0;
  int64_t requested_size = 0;
  int32_t gc_counter = 0;
  bool allocated = false;
  bool active = false;
  std::shared_ptr<GatheredContext>
      context_when_allocated;  // per-watcher context
};

// Struct containing info of a memory segment (i.e. one contiguous
// mallocDevice).
struct SegmentInfo {
  int64_t device = 0;
  int64_t address = 0;
  int64_t total_size = 0;
  int64_t requested_size = 0;  // unrounded, actually requested size
  int64_t allocated_size = 0;
  int64_t active_size = 0;
  deviceStream_t stream = 0;
  bool is_large = false;
  bool is_expandable = false;
  std::vector<BlockInfo> blocks;
  std::shared_ptr<GatheredContext> context_when_allocated;
};

struct AllocatorState {
  virtual ~AllocatorState() = default;
};

struct TraceEntry {
  enum Action {
    ALLOC,           // API made to the caching allocator for new memory
    FREE_REQUESTED,  // API call made to the caching allocator to free memory
    FREE_COMPLETED,  // The allocator might have to delay a free because
                     // it is still in use on another stream via record_stream
                     // This event is generated when a free actually completes.
    SEGMENT_ALLOC,   // a call to mallocDevice to get more memory from the OS
    SEGMENT_FREE,    // a call to freeDevice to return memory to the OS (e.g. to
                     // defragment or empty_caches)
    SEGMENT_MAP,     // a call to deviceMemMap (used with expandable_segments)
    SEGMENT_UNMAP,   // unmap part of a segment (used with expandable segments)
    SNAPSHOT,  // a call to snapshot, used to correlate memory snapshots to
               // trace events
    OOM  // the allocator threw an OutOfMemoryError (addr_ is the amount of free
         // bytes reported by device)
  };
  TraceEntry(Action action, int64_t addr, size_t size, deviceStream_t stream,
             std::shared_ptr<GatheredContext> context = nullptr)
      : action_(action),
        addr_(addr),
        context_(std::move(context)),
        stream_(stream),
        size_(size) {}
  Action action_;
  int64_t
      addr_;  // for OOM, this is the amount of free bytes reported by device
  std::shared_ptr<GatheredContext> context_;
  deviceStream_t stream_;
  size_t size_;
};

struct SnapshotInfo {
  std::vector<SegmentInfo> segments;
  std::vector<std::vector<TraceEntry>> device_traces;
};

enum struct RecordContext {
  NEVER = 0,
  STATE = 1,  // only keep stacks for active allocations
  ALLOC = 2,  // additionally keep stacks for allocations in the trace history
  ALL = 3,    // additionally record stacks for when something is freed
};

void setAllocatorSettings(const std::string& env);

// Size pretty-printer
std::string format_size(uint64_t size);

using OutOfMemoryObserver =
    std::function<void(int64_t device, int64_t allocated, int64_t device_total,
                       int64_t device_free)>;

class DeviceAllocator : public c10::Allocator {
 public:
  virtual void* raw_alloc(size_t nbytes) = 0;
  virtual void* raw_alloc_with_stream(size_t nbytes, deviceStream_t stream) = 0;
  virtual void raw_delete(void* ptr) = 0;
  virtual void init(int device_count) = 0;
  virtual bool initialized() = 0;
  virtual void setMemoryFraction(double fraction, int device) = 0;
  virtual void emptyCache() = 0;
  virtual void cacheInfo(int dev_id, size_t* largestBlock) = 0;
  virtual void* getBaseAllocation(void* ptr, size_t* size) = 0;
  virtual void recordStream(const c10::DataPtr&, const DIPUStream& stream) = 0;
  virtual DeviceStats getDeviceStats(int device) = 0;
  virtual void resetAccumulatedStats(int device) = 0;
  virtual void resetPeakStats(int device) = 0;
  virtual SnapshotInfo snapshot() = 0;
  virtual bool isHistoryEnabled() {
    TORCH_CHECK(
        false, name(),
        " does not yet support recordHistory. "
        "If you need it, please file an issue describing your use case.");
  }
  virtual void recordHistory(bool enabled, CreateContextFn context_recorder,
                             size_t alloc_trace_max_entries,
                             RecordContext when) = 0;
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) = 0;
  virtual std::string name() = 0;
};

// Allocator object, statically initialized
// See BackendInitializer in CUDACachingAllocator.cpp.
// Atomic loads on x86 are just normal loads,
// (atomic stores are different), so reading this value
// is no different than loading a pointer.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern std::atomic<DeviceAllocator*> allocator;

inline DeviceAllocator* getTorchAllocator() { return allocator.load(); }

// Called directly by clients.
inline void* raw_alloc(size_t nbytes) {
  return getTorchAllocator()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, deviceStream_t stream) {
  return getTorchAllocator()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  return getTorchAllocator()->raw_delete(ptr);
}

inline void init(int device_count) {
  return getTorchAllocator()->init(device_count);
}

inline void setMemoryFraction(double fraction, int device) {
  return getTorchAllocator()->setMemoryFraction(fraction, device);
}

inline void emptyCache() { return getTorchAllocator()->emptyCache(); }

inline void cacheInfo(int dev_id, size_t* largestBlock) {
  return getTorchAllocator()->cacheInfo(dev_id, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return getTorchAllocator()->getBaseAllocation(ptr, size);
}

inline DeviceStats getDeviceStats(int device) {
  return getTorchAllocator()->getDeviceStats(device);
}

inline void resetAccumulatedStats(int device) {
  return getTorchAllocator()->resetAccumulatedStats(device);
}

inline void resetPeakStats(int device) {
  return getTorchAllocator()->resetPeakStats(device);
}

inline SnapshotInfo snapshot() { return getTorchAllocator()->snapshot(); }

inline void recordHistory(bool enabled, CreateContextFn context_recorder,
                          size_t alloc_trace_max_entries, RecordContext when) {
  return getTorchAllocator()->recordHistory(enabled, context_recorder,
                                            alloc_trace_max_entries, when);
}

inline bool isHistoryEnabled() {
  return getTorchAllocator()->isHistoryEnabled();
}

inline void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
  return getTorchAllocator()->attachOutOfMemoryObserver(std::move(observer));
}

inline std::string name() { return getTorchAllocator()->name(); }

inline void recordStream(const c10::DataPtr& ptr, const DIPUStream& stream) {
  getTorchAllocator()->recordStream(ptr, stream);
}

}  // namespace dipu::allocator
