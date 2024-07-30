// Copyright (c) 2024, DeepLink.
#include "DIPUCachingDeviceAllocator.h"

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>

#include "csrc_dipu/base/environ.hpp"
#include "csrc_dipu/runtime/core/DIPUEvent.h"
#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

#include "DIPUCachingAllocator.h"
#include "ExpandableSegment.h"
#include "allocator_metrics.h"

// NOLINTBEGIN
// ----------------------------------------------------------------------------
// Code from pytorch2.1.0 c10/cuda/CUDACachingAllocator.cpp
// Our changes:
//    1. use dipu runtime api to replace CUDA API.
//    2. make ExpandableSegment to an interface class, vendor should implement
//    it if want to support expandable segments. If
//    vendorCreateExpandableSegment is not implemented, the allocator will never
//    use expandable segments.
//    3. remove EventPool class, DIPU already supports it.
// ----------------------------------------------------------------------------
namespace dipu::allocator {

C10_DEFINE_REGISTRY(FreeDeviceMemoryCallbacksRegistry, FreeMemoryCallback);

//
// Yet another caching allocator for device allocations.
//
// - Allocations are associated with a stream. Once freed, blocks can be
//   re-allocated on the same stream, but not on any other stream.
// - The allocator attempts to find the smallest cached block that will fit the
//   requested size. If the block is larger than the requested size, it may be
//   split. If no block is found, the allocator will delegate to mallocDevice.
// - If the mallocDevice fails, the allocator will attempt to free one cached
//   block of sufficient size that is not split and retry the allocation.
//   If this also fails, the allocator will attempt to free all cached blocks
//   that are not split and retry the allocation.
// - Large (>1MB) and small allocations are stored in separate pools.
//   Small requests are packed into 2MB buffers. Large requests will use the
//   smallest available free block or allocate a new block using mallocDevice.
// - To reduce fragmentation, requests between 1MB and 10MB will allocate and
//   split a 20MB block, if no free block of sufficient size is available.
// - To further reduce fragmentation, blocks >= 200MB are not allowed to be
//   split. These oversize cached blocks will still satisfy requests within
//   20MB of the oversize cached block size.
//
// With this allocator, allocations and frees should logically be considered
// "usages" of the memory segment associated with streams, just like kernel
// launches. The programmer must insert the proper synchronization if memory
// segments are used from multiple streams.
//
// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the block is not reused before each recorded stream completes
// work.
//

constexpr size_t kMinBlockSize =
    512;  // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576;  // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152;  // "small" allocations are packed in 2 MiB blocks
constexpr size_t kLargeBuffer =
    20971520;  // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760;  // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152;  // round up large allocations to 2 MiB
constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

namespace {

using stream_set = ska::flat_hash_set<DIPUStream>;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in DEVICE allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) { stat.peak = stat.current; }

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(StatArray& stat_array, int64_t amount,
                       const StatTypes& stat_types) {
  for_each_selected_stat_type(stat_types,
                              [&stat_array, amount](size_t stat_type) {
                                update_stat(stat_array[stat_type], amount);
                              });
}

struct Block;
using Comparison = bool (*)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);
static bool BlockComparatorAddress(const Block* a, const Block* b);

struct BlockPool {
  explicit BlockPool(bool small)
      : blocks(BlockComparatorSize),
        unmapped(BlockComparatorAddress),
        is_small(small) {}
  std::set<Block*, Comparison> blocks;
  std::set<Block*, Comparison> unmapped;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool is_small;
};

struct Block {
  int device;                // gpu
  deviceStream_t stream;     // allocation stream
  stream_set stream_uses;    // streams on which the block was used
  size_t size;               // block size in bytes
  size_t requested_size;     // memory originally requested
  BlockPool* pool{nullptr};  // owning memory pool
  void* ptr{nullptr};        // memory address
  bool allocated{false};     // in-use flag
  bool mapped{true};     // is the virtual address range this Block references
                         // backed by physical pages. Always true when
                         // expandable_segment_ is null. When false
                         // This Block will be aligned to the segment size
                         // of its expandable_segment_.
  Block* prev{nullptr};  // prev block if split from a larger allocation
  Block* next{nullptr};  // next block if split from a larger allocation
  int event_count{0};    // number of outstanding DEVICE events
  int gc_count{0};  // counter for prioritizing older / less useful blocks for
                    // garbage collection
  std::shared_ptr<GatheredContext> context_when_allocated;
  // only set for the first block in the segment (when prev == null)
  // this records the frame information when mallocDevice was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<GatheredContext> context_when_segment_allocated;

  ExpandableSegment* expandable_segment_{nullptr};

  Block(int device, deviceStream_t stream, size_t size, BlockPool* pool,
        void* ptr)
      : device(device),
        stream(stream),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(int device, deviceStream_t stream, size_t size)
      : device(device), stream(stream), size(size), requested_size(0) {}

  bool is_split() const { return (prev != nullptr) || (next != nullptr); }
  void splice(Block* before, Block* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
           reinterpret_cast<uintptr_t>(b->stream);
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
         reinterpret_cast<uintptr_t>(b->ptr);
}

bool BlockComparatorAddress(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return reinterpret_cast<uintptr_t>(a->stream) <
           reinterpret_cast<uintptr_t>(b->stream);
  }
  return reinterpret_cast<uintptr_t>(a->ptr) <
         reinterpret_cast<uintptr_t>(b->ptr);
}

struct AllocParams {
  AllocParams(int device, size_t size, deviceStream_t stream, BlockPool* pool,
              size_t alloc_size, DeviceStats& stats)
      : search_key(device, stream, size), pool(pool), alloc_size(alloc_size) {}

  int device() const { return search_key.device; }
  deviceStream_t stream() const { return search_key.stream; }
  size_t size() const { return search_key.size; }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block{};
  StatTypes stat_types = {false};
  devapis::OpStatus err{};
};

}  // anonymous namespace

// Environment config parser
// Defined here, rather than its own .cpp file,
// because parseArgs needs to know kLargeBuffer.
// Defined outside namespace Native because it's not Native-specific.
class CachingAllocatorConfig {
 public:
  static size_t max_split_size() { return instance().m_max_split_size; }

  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() { return instance().m_expandable_segments; }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As ane example, if we want 4 divisions between 2's power, this can be done
  // using env variable: DIPU_TORCH_ALLOCATOR_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size) {
    size_t log_size = (63 - c10::llvm::countLeadingZeros(size));

    // Our intervals start at 1MB and end at 64GB
    const size_t interval_start =
        63 - c10::llvm::countLeadingZeros(static_cast<size_t>(1048576));
    const size_t interval_end =
        63 - c10::llvm::countLeadingZeros(static_cast<size_t>(68719476736));
    TORCH_CHECK((interval_end - interval_start == kRoundUpPowerOfTwoIntervals),
                "kRoundUpPowerOfTwoIntervals mismatch");

    int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

    index = std::max(0, index);
    index = std::min(index, static_cast<int>(kRoundUpPowerOfTwoIntervals) - 1);
    return instance().m_roundup_power2_divisions[index];
  }

  static CachingAllocatorConfig& instance() {
    static CachingAllocatorConfig* s_instance = ([]() {
      auto inst = new CachingAllocatorConfig();
      inst->parseArgs(environ::torchAllocatorConf().c_str());
      return inst;
    })();
    return *s_instance;
  }

  void parseArgs(const char* env);

 private:
  CachingAllocatorConfig()
      : m_max_split_size(std::numeric_limits<size_t>::max()),
        m_garbage_collection_threshold(0),
        m_expandable_segments(false) {
    m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  }

  static void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(const std::vector<std::string>& config, size_t i, char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(const std::vector<std::string>& config,
                                         size_t i);
  size_t parseRoundUpPower2Divisions(const std::vector<std::string>& config,
                                     size_t i);

  std::atomic<size_t> m_max_split_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<bool> m_expandable_segments;
};

void CachingAllocatorConfig::lexArgs(const char* env,
                                     std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void CachingAllocatorConfig::consumeToken(
    const std::vector<std::string>& config, size_t i, char c) {
  TORCH_CHECK(i < config.size() && config[i] == std::string(1, c),
              "Error parsing CachingAllocator settings, expected ", c, "");
}

size_t CachingAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / (1024 * 1024),
        "CachingAllocator option max_split_size_mb too small, must be > ",
        kLargeBuffer / (1024 * 1024), "");
    val1 = std::max(val1, kLargeBuffer / (1024 * 1024));
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / (1024 * 1024)));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(val1 > 0, "garbage_collect_threshold too small, set it 0.0~1.0",
                "");
    TORCH_CHECK(val1 < 1.0, "garbage_collect_threshold too big, set it 0.0~1.0",
                "");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(false, "Error, expecting garbage_collection_threshold value",
                "");
  }
  return i;
}

size_t CachingAllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config, size_t i) {
  consumeToken(config, ++i, ':');
  bool first_value = true;

  if (++i < config.size()) {
    if (config[i] == "[") {
      size_t last_index = 0;
      while (++i < config.size() && config[i] != "]") {
        const std::string& val1 = config[i];
        size_t val2 = 0;

        consumeToken(config, ++i, ':');
        if (++i < config.size()) {
          val2 = stoi(config[i]);
        } else {
          TORCH_CHECK(false, "Error parsing roundup_power2_divisions value",
                      "");
        }
        TORCH_CHECK(c10::llvm::isPowerOf2_64(val2),
                    "For roundups, the divisons has to be power of 2 ", "");

        if (val1 == ">") {
          std::fill(
              std::next(m_roundup_power2_divisions.begin(),
                        static_cast<std::vector<uint64_t>::difference_type>(
                            last_index)),
              m_roundup_power2_divisions.end(), val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(c10::llvm::isPowerOf2_64(val1_long),
                      "For roundups, the intervals have to be power of 2 ", "");

          size_t index = 63 - c10::llvm::countLeadingZeros(val1_long);
          index = std::max(static_cast<size_t>(0), index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          if (first_value) {
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<uint64_t>::difference_type>(index)),
                val2);
            first_value = false;
          }
          if (index < m_roundup_power2_divisions.size()) {
            m_roundup_power2_divisions[index] = val2;
          }
          last_index = index;
        }

        if (config[i + 1] != "]") {
          consumeToken(config, ++i, ',');
        }
      }
    } else {  // Keep this for backwards compatibility
      size_t val1 = stoi(config[i]);
      TORCH_CHECK(c10::llvm::isPowerOf2_64(val1),
                  "For roundups, the divisons has to be power of 2 ", "");
      std::fill(m_roundup_power2_divisions.begin(),
                m_roundup_power2_divisions.end(), val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  return i;
}

void CachingAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;

  if (env == nullptr) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    if (config[i] == "max_split_size_mb") {
      i = parseMaxSplitSize(config, i);
    } else if (config[i] == "garbage_collection_threshold") {
      i = parseGarbageCollectionThreshold(config, i);
    } else if (config[i] == "roundup_power2_divisions") {
      i = parseRoundUpPower2Divisions(config, i);
    } else if (config[i] == "expandable_segments") {
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() && (config[i] == "True" || config[i] == "False"),
          "Expected a single True/False argument for expandable_segments");
      m_expandable_segments = (config[i] == "True");
      if (m_expandable_segments && !vendorCreateExpandableSegment) {
        DIPU_LOGW(
            "expandable_segments is set to True, but no implementation of "
            "vendorCreateExpandableSegment is found. Hence ignoring the "
            "setting.");
        m_expandable_segments = false;
      }
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }
}

static std::string reportProcessMemoryInfo(int device) { return ""; }

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // device statistics
  DeviceStats stats;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;

  // outstanding cuda events
  ska::flat_hash_map<DIPUStream, std::deque<std::pair<DIPUEvent, Block*>>>
      events;

  // record used memory.
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment*> expandable_segments_;

  bool set_fraction = false;

  bool record_history = false;
  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  RecordContext record_context_ = RecordContext::NEVER;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry>*
      alloc_trace;  // pointer because we need to intentionally leak this on
                    // deallocation it can hold references to Python state which
                    // will already be destroyed when we are in exit handlers

  // XXX - maybe we should generalize and have multiple events
  std::vector<OutOfMemoryObserver> oom_observers_;

 public:
  DeviceCachingAllocator()
      : large_blocks(/*small=*/false),
        small_blocks(/*small=*/true),
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = CachingAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);
  }

  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries, RecordContext when) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(when == RecordContext::NEVER || context_recorder);
    record_history = enabled;
    context_recorder_.store(record_history ? context_recorder : nullptr);
    alloc_trace_max_entries_ =
        std::max(static_cast<size_t>(1), alloc_trace_max_entries);
    record_context_ = enabled ? when : RecordContext::NEVER;
    alloc_trace_next = 0;
    alloc_trace->clear();
  }

  bool isHistoryEnabled() const { return record_history; }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) {
    oom_observers_.emplace_back(std::move(observer));
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<GatheredContext> maybeGatherContext(RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.
  Block* malloc(int device, size_t orig_size, deviceStream_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);
    if (device < 0) {
      device = devproxy::current_device();
    }

    std::unique_lock<std::recursive_mutex> lock(mutex);
    process_events(context);

    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && get_free_block(params));

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(set_fraction &&
                       CachingAllocatorConfig::garbage_collection_threshold() >
                           0.0)) {
        garbage_collect_cached_blocks();
      }
      // Attempt allocate
      block_found = alloc_block(params, false, context)
                    // Free enough available cached blocks to satisfy alloc and
                    // retry alloc.
                    || (release_available_cached_blocks(params) &&
                        alloc_block(params, false, context))
                    // Free all non-split cached blocks and retry alloc.
                    || (release_cached_blocks(context) &&
                        alloc_block(params, true, context));
    }

    if (!block_found) {
      // For any error code other than cudaErrorMemoryAllocation,
      // alloc_block should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == devapis::OpStatus::ERR_NOMEM);

      devapis::DIPUDeviceStatus dev_status = devproxy::getDeviceStatus(device);
      size_t device_free = dev_status.freeGlobalMem;
      size_t device_total = dev_status.totalGlobalMem;
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      std::string proc_info = reportProcessMemoryInfo(device);

      if (record_history) {
        record_trace(TraceEntry::OOM, device_free, params.size(),
                     params.stream(), std::move(context));
      }
      stats.num_ooms += 1;

      c10::reportOutOfMemoryToProfiler(
          size,
          stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          c10::Device(c10::DeviceType::CUDA,
                      static_cast<c10::DeviceIndex>(device)));

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto observers_local = oom_observers_;

      // Make sure we do not have the device lock before calling our
      // observers which might need hold the GIL
      // It is safe to release at this point because will no longer
      // be reading any allocator state.

      lock.unlock();

      for (const auto& obs : observers_local) {
        obs(device, alloc_size,
            set_fraction ? allowed_memory_maximum : device_total, device_free);
      }

      // "total capacity": total global memory on GPU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CUDA API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_blocks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger block that is partially in-use.
      TORCH_CHECK_WITH(
          OutOfMemoryError, false, "CUDA out of memory. Tried to allocate ",
          format_size(alloc_size), ". GPU ", device, " has a total capacty of ",
          format_size(device_total), " of which ", format_size(device_free),
          " is free. ", proc_info, "Of the allocated memory ",
          format_size(allocated_bytes), " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " If reserved but unallocated memory is large try setting "
          "max_split_size_mb to avoid"
          " fragmentation.  See documentation for Memory Management and "
          "DIPU_TORCH_ALLOCATOR_CONF",
          "");
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(params, orig_size, std::move(context),
                             split_remainder);
  }

  Block* alloc_found_block(const AllocParams& params, size_t orig_size,
                           std::shared_ptr<GatheredContext> context,
                           bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(params.err == devapis::OpStatus::SUCCESS &&
                          params.block != nullptr &&
                          params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    const bool already_split = block->is_split();
    if (split_remainder) {
      remaining = block;

      block = new Block(device, stream, size, pool, block->ptr);
      block->expandable_segment_ = remaining->expandable_segment_;
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      bool inserted = pool->blocks.insert(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (already_split && !block->expandable_segment_) {
        // An already-split inactive block is being shrunk by size bytes.
        update_stat_array(stats.inactive_split_bytes,
                          -static_cast<std::int64_t>(block->size),
                          params.stat_types);
      } else if (!block->expandable_segment_) {
        // A new split inactive block is being created from a previously unsplit
        // block, size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(stats.inactive_split_bytes[stat_type],
                      static_cast<std::int64_t>(remaining->size));
          update_stat(stats.inactive_split[stat_type], 1);
        });
      }

    } else if (already_split && !block->expandable_segment_) {
      // An already-split block is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(stats.inactive_split_bytes[stat_type],
                    -static_cast<std::int64_t>(block->size));
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    block->allocated = true;
    block->requested_size = orig_size;
    if (record_history) {
      block->context_when_allocated = std::move(context);
      record_trace(TraceEntry::ALLOC, int64_t(block->ptr), orig_size,
                   block->stream, block->context_when_allocated);
    }

    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(stats.allocated_bytes[stat_type],
                  static_cast<std::int64_t>(block->size));
      update_stat(stats.active[stat_type], 1);
      update_stat(stats.active_bytes[stat_type],
                  static_cast<std::int64_t>(block->size));
      update_stat(stats.requested_bytes[stat_type],
                  static_cast<std::int64_t>(block->requested_size));
    });
    if (block->size >= CachingAllocatorConfig::max_split_size()) {
      update_stat(stats.oversize_allocations, 1);
    }

    c10::reportMemoryUsageToProfiler(
        block->ptr, block->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, device));

    return block;
  }

  void free(Block* block) {
    std::shared_ptr<GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    // following logic might modifying underlaying Block, causing the size
    // changed. We store ahead for reporting
    auto orig_block_ptr = block->ptr;
    auto orig_block_size = block->size;

    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(stats.allocated_bytes[stat_type],
                  -static_cast<std::int64_t>(block->size));
    });
    if (record_history) {
      record_trace(TraceEntry::FREE_REQUESTED,
                   reinterpret_cast<int64_t>(block->ptr), block->requested_size,
                   block->stream,
                   context ? context : block->context_when_allocated);
    }
    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!block->stream_uses.empty()) {
      insert_events(block);
    } else {
      free_block(block, context);
    }

    c10::reportMemoryUsageToProfiler(
        orig_block_ptr, -orig_block_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::CUDA, block->device));
  }

  void* getBaseAllocation(Block* block, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(!block->expandable_segment_,
                "Tensors allocated with expandable_segments:True cannot be "
                "shared between processes. Consider using "
                "expandable_segments:False in data loading workers via "
                "torch.cuda.memory._set_allocator_settings('expandable_"
                "segments:False')");
    while (block->prev) {
      block = block->prev;
    }
    void* basePtr = block->ptr;
    if (outSize) {
      size_t size = 0;
      while (block) {
        size += block->size;
        block = block->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(Block* block, const DIPUStream& stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.rawstream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    devapis::DIPUDeviceStatus dev_status =
        devproxy::getDeviceStatus(devproxy::current_device());
    allowed_memory_maximum =
        static_cast<size_t>(fraction * dev_status.totalGlobalMem);
    set_fraction = true;
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache() {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(context);
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) {  // make an initial guess if a zero *largest is passed in
      devapis::DIPUDeviceStatus dev_status =
          devproxy::getDeviceStatus(devproxy::current_device());
      *largest = dev_status.freeGlobalMem;
    }
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
  }

  /** Returns a copy of the memory allocator stats **/
  DeviceStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    size_t total_active = 0;
    std::vector<SegmentInfo> result;
    const auto all_blocks = get_all_blocks();

    for (const Block* const head_block : all_blocks) {
      // For expandable segments, we report one segment for each continguous
      // mapped range of memory
      if (head_block->prev && head_block->prev->mapped) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_block->device;
      segment_info.address = reinterpret_cast<int64_t>(head_block->ptr);
      segment_info.stream = head_block->stream;
      segment_info.is_large = (!head_block->pool->is_small);
      segment_info.is_expandable = head_block->expandable_segment_;
      segment_info.context_when_allocated =
          head_block->context_when_segment_allocated;

      const Block* block = head_block;
      while (block != nullptr && block->mapped) {
        segment_info.blocks.emplace_back();
        BlockInfo& block_info = segment_info.blocks.back();

        block_info.size = block->size;
        block_info.requested_size = block->requested_size;
        block_info.allocated = block->allocated;
        block_info.active = block->allocated || (block->event_count > 0) ||
                            !block->stream_uses.empty();

        segment_info.total_size += block_info.size;
        if (block_info.allocated) {
          segment_info.allocated_size += block_info.size;
        }
        if (block_info.active) {
          segment_info.active_size += block_info.size;
          segment_info.requested_size += block_info.requested_size;
        }
        block_info.context_when_allocated = block->context_when_allocated;
        block = block->next;
      }
      total_active += segment_info.active_size;
    }

    std::sort(result.begin(), result.end(),
              [](const SegmentInfo& a, const SegmentInfo& b) {
                return a.address < b.address;
              });

    if (record_history) {
      record_trace(TraceEntry::SNAPSHOT, 0, total_active, nullptr, nullptr);
    }
    return result;
  }

  std::vector<TraceEntry> trace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    result.reserve(alloc_trace->size());
    result.insert(result.end(), alloc_trace->begin() + alloc_trace_next,
                  alloc_trace->end());
    result.insert(result.end(), alloc_trace->begin(),
                  alloc_trace->begin() + alloc_trace_next);
    return result;
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (c10::llvm::isPowerOf2_64(size)) {
      return size;
    }

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = c10::llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - c10::llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

  static size_t round_size(size_t size) {
    size += getMemoryAlignmentStrategy()->getBeta();
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    }
    auto divisions = CachingAllocatorConfig::roundup_power2_divisions(size);
    if (divisions > 0 && size > (kMinBlockSize * divisions)) {
      return roundup_power2_next_division(size, divisions);
    }
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
  }

  bool hasAllocatedExpandableSegments() const {
    return !expandable_segments_.empty();
  }

 private:
  // All private methods do not acquire the allocator mutex.
  std::vector<const Block*> get_all_blocks() const {
    std::vector<const Block*> blocks;
    blocks.insert(blocks.end(), small_blocks.blocks.begin(),
                  small_blocks.blocks.end());
    blocks.insert(blocks.end(), large_blocks.blocks.begin(),
                  large_blocks.blocks.end());
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  // returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
  Block* find_expandable_block(int device, deviceStream_t stream,
                               BlockPool* pool, size_t size) {
    Block key(device, stream, 0);

    auto allocatable = [](Block* b) {
      return b && !b->allocated && b->event_count == 0 &&
             b->stream_uses.empty();
    };
    auto has_available_address_space = [&](Block* b) {
      size_t bytes = 0;
      while (bytes < size && allocatable(b)) {
        bytes += b->size;
        b = b->next;
      }
      return bytes >= size;
    };
    for (auto it = pool->unmapped.lower_bound(&key);
         it != pool->unmapped.end() && (*it)->stream == stream; ++it) {
      Block* c = *it;
      // we found the lowest address of an unmapped segment
      // but there might be a free segment we can also use
      // right before it
      if (allocatable(c->prev)) {
        c = c->prev;
      }
      if (has_available_address_space(c)) {
        return c;
      }
    }
    auto segment_size = pool->is_small ? kSmallBuffer : kLargeBuffer;
    expandable_segments_.emplace_back(
        createExpandableSegment(device, stream, segment_size));

    ExpandableSegment* es = expandable_segments_.back();
    Block* candidate = new Block(device, stream, es->size(), pool, es->ptr());
    candidate->mapped = false;
    candidate->expandable_segment_ = es;
    pool->unmapped.insert(candidate);
    return candidate;
  }

  bool map_block(Block* to_map, size_t size,
                 const std::shared_ptr<GatheredContext>& ctx) {
    TORCH_INTERNAL_ASSERT(!to_map->mapped && size <= to_map->size);
    TORCH_INTERNAL_ASSERT(
        !to_map->context_when_allocated);  // unmapped blocks should not keep
                                           // history
    auto mapped_range =
        to_map->expandable_segment_->map(SegmentRange{to_map->ptr, size});
    // failed to map the memory
    if (mapped_range.size == 0) {
      return false;
    }
    TORCH_INTERNAL_ASSERT(mapped_range.ptr == to_map->ptr &&
                          mapped_range.size >= size);

    BlockPool& pool = *to_map->pool;
    pool.unmapped.erase(to_map);
    to_map->mapped = true;

    if (mapped_range.size < to_map->size) {
      // to_map -> remaining -> to_map->next(?)
      Block* remaining = new Block(
          to_map->device, to_map->stream, to_map->size - mapped_range.size,
          &pool, static_cast<char*>(to_map->ptr) + mapped_range.size);
      remaining->mapped = false;
      remaining->expandable_segment_ = to_map->expandable_segment_;
      remaining->splice(to_map, to_map->next);
      pool.unmapped.insert(remaining);
      to_map->size = mapped_range.size;
    }

    try_merge_blocks(to_map, to_map->prev, pool);
    try_merge_blocks(to_map, to_map->next, pool);

    pool.blocks.insert(to_map);

    // update statistics
    total_allocated_memory += mapped_range.size;
    StatTypes stat_types = get_stat_types_for_pool(*to_map->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], mapped_range.size);
    });
    if (record_history) {
      record_trace(TraceEntry::SEGMENT_MAP, int64_t(mapped_range.ptr),
                   mapped_range.size, to_map->stream, ctx);
      if (!to_map->prev && !to_map->context_when_segment_allocated) {
        to_map->context_when_segment_allocated = ctx;
      }
    }

    return true;
  }

  Block* try_allocate_expandable_block(
      int device, deviceStream_t stream, BlockPool* pool, size_t size,
      const std::shared_ptr<GatheredContext>& ctx) {
    Block* candidate = find_expandable_block(device, stream, pool, size);
    // Candidate is now a list free/unmapped blocks with at least size room:
    // unmapped -> null
    // unmapped -> free -> *
    // free -> unmapped -> *

    if (!candidate->mapped &&
        !map_block(candidate, std::min(candidate->size, size), ctx)) {
      return nullptr;
    }
    TORCH_INTERNAL_ASSERT(candidate->mapped);

    while (candidate->size < size) {
      // invariant: free -> unmapped -> *
      // map_block will map some of unmapped and merge with free
      auto remaining = size - candidate->size;
      auto new_candidate = candidate->next;
      if (!map_block(new_candidate, std::min(remaining, candidate->next->size),
                     ctx)) {
        return nullptr;
      }
      candidate = new_candidate;
    }
    pool->blocks.erase(candidate);
    return candidate;
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block,
                  const std::shared_ptr<GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(!block->allocated && block->event_count == 0 &&
                          block->stream_uses.empty());
    if (record_history) {
      record_trace(TraceEntry::FREE_COMPLETED, int64_t(block->ptr),
                   block->requested_size, block->stream,
                   context ? context : block->context_when_allocated);
    }
    block->context_when_allocated = nullptr;
    size_t original_block_size = block->size;
    size_t requested_size = block->requested_size;

    auto& pool = *block->pool;
    int64_t net_change_inactive_split_blocks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      const int64_t subsumed_size =
          try_merge_blocks(block, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_blocks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (block->is_split()) {
      net_change_inactive_split_blocks += 1;
      net_change_inactive_split_size += block->size;
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);

    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that blocks
      // cannot be freed when requested, but fully free pages
      // of expandable blocks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segements from
      // inactive_split
      if (!block->expandable_segment_) {
        update_stat(stats.inactive_split[stat_type],
                    net_change_inactive_split_blocks);
        update_stat(stats.inactive_split_bytes[stat_type],
                    net_change_inactive_split_size);
      }
      update_stat(stats.active[stat_type], -1);
      update_stat(stats.active_bytes[stat_type],
                  -static_cast<std::int64_t>(original_block_size));
      update_stat(stats.requested_bytes[stat_type],
                  -static_cast<std::int64_t>(requested_size));
    });
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) {  // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      dst->context_when_segment_allocated =
          std::move(src->context_when_segment_allocated);
    } else {  // [dest src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased =
        src->mapped ? pool.blocks.erase(src) : pool.unmapped.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size, deviceStream_t stream) {
    if (size <= kSmallSize) {
      return small_blocks;
    }
    return large_blocks;
  }

  static StatTypes get_stat_types_for_pool(const BlockPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  static bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small ||
        CachingAllocatorConfig::expandable_segments()) {
      return remaining >= kMinBlockSize;
    }
    return (size < CachingAllocatorConfig::max_split_size()) &&
           (remaining > kSmallSize);
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    }
    if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    }
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
  }

  bool get_free_block(AllocParams& p) const {
    BlockPool& pool = *p.pool;

    if (C10_UNLIKELY(set_fraction &&
                     CachingAllocatorConfig::garbage_collection_threshold() >
                         0.0)) {
      // Track block reuse interval only when garbage collection is enabled.
      for (auto& b : pool.blocks) {
        ++b->gc_count;
      }
    }
    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) return false;

    if ((*it)->expandable_segment_) {
      if (CachingAllocatorConfig::expandable_segments()) {
        // if we are allocated to the part of the block that is expandable
        // for the purposes of "best fit" we consider its size to be the size it
        // can expand to, not the size it currently is. This means that we
        // sometimes have to search for blocks with bigger 'size' before
        // choosing this segment.
        auto expandable_size = [](Block* b) {
          return b->size + (b->next && !b->next->mapped ? b->next->size : 0);
        };
        auto next = it;
        next++;
        while ((*it)->expandable_segment_ && next != pool.blocks.end() &&
               (*next)->stream == p.stream() &&
               expandable_size(*next) < expandable_size(*it)) {
          it = next++;
        }
      } else {
        // Rarely expandable segments has been turned off after we have
        // already allocated some blocks as expandable. For instance,
        // since we cannot share expandable memory via IPC, someone might
        // temporarily disable it. In this case we need to honor this request
        // by only finding non-expandable blocks
        do {
          it++;
        } while (it != pool.blocks.end() && (*it)->expandable_segment_ &&
                 (*it)->stream == p.stream());
        if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
          return false;
        }
      }
    }

    // Do not return an oversized block for a large request
    if ((p.size() < CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= CachingAllocatorConfig::max_split_size())) {
      return false;
    }
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= CachingAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + kLargeBuffer)) {
      return false;
    }
    p.block = *it;
    (*it)->gc_count = 0;  // Denote this block has been used
    pool.blocks.erase(it);
    return true;
  }

  static bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeDeviceMemoryCallbacksRegistry()->Keys()) {
      freed_memory |=
          FreeDeviceMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_blocks() {
    // Free unused cached blocks to reclaim GPU memory.
    // Unlike release_cached_blocks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    auto gc_threshold = static_cast<size_t>(
        CachingAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able blocks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_block_count = 0;
    for (auto& b : large_blocks.blocks) {
      if (!b->is_split()) {
        total_age += b->gc_count;
        ++freeable_block_count;
      }
    }
    // No free-able blocks?
    if (freeable_block_count == 0) {
      return;
    }

    // TODO(caikun): will sync device in NPUCachingAllocator
    // npuSynchronizeDevice();

    // Repeat GC until we reach reclaim > target size.
    bool block_freed = true;
    while (gc_reclaimed < target_size && block_freed &&
           freeable_block_count > 0) {
      // Free blocks exceeding this age threshold first.
      double age_threshold = total_age / freeable_block_count;
      // Stop iteration if we can no longer free a block.
      block_freed = false;

      // Free blocks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_blocks.blocks.begin();
      while (it != large_blocks.blocks.end()) {
        Block* block = *it;
        ++it;
        if (!block->is_split() && block->gc_count >= age_threshold) {
          block_freed = true;
          gc_reclaimed += block->size;
          total_age -= block->gc_count;  // Decrement the age
          freeable_block_count--;        // One less block that can be freed
          release_block(block);
        }
      }
    }
  }

  bool alloc_block(AllocParams& p, bool isRetry,
                   const std::shared_ptr<GatheredContext>& ctx) {
    // Defensively checks for preexisting CUDA error state.
    // C10_CUDA_CHECK(cudaGetLastError());
    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + size > allowed_memory_maximum) {
      p.err = devapis::OpStatus::ERR_NOMEM;
      return false;
    }

    if (CachingAllocatorConfig::expandable_segments()) {
      p.block = try_allocate_expandable_block(p.device(), p.stream(), p.pool,
                                              p.size(), ctx);
      if (p.block != nullptr) {
        p.err = devapis::OpStatus::SUCCESS;
      } else {
        p.err = devapis::OpStatus::ERR_NOMEM;
      }
      return p.block != nullptr;
    }

    p.err = devproxy::mallocDevice(&ptr, size, false);
    if (p.err != devapis::OpStatus::SUCCESS) {
      // We can only handle NOMEM errors, and throw exceptions when
      // encountering other errors
      TORCH_CHECK(p.err == devapis::OpStatus::ERR_NOMEM,
                  "device error, p.err = ", static_cast<int32_t>(p.err));
      return false;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool,
                        static_cast<char*>(ptr));
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], size);
    });
    if (size >= CachingAllocatorConfig::max_split_size()) {
      update_stat(stats.oversize_segments, 1);
    }

    // p.block came from new, not cudaMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.block != nullptr && p.block->ptr != nullptr);
    if (record_history) {
      record_trace(TraceEntry::SEGMENT_ALLOC,
                   reinterpret_cast<int64_t>(p.block->ptr), p.block->size,
                   p.stream(), ctx);
      p.block->context_when_segment_allocated = ctx;
    }
    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (CachingAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max()) {
      return false;
    }
    BlockPool& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    Block key(p.search_key.device, p.search_key.stream, p.search_key.size,
              p.search_key.pool, p.search_key.ptr);
    key.size = (key.size < CachingAllocatorConfig::max_split_size())
                   ? CachingAllocatorConfig::max_split_size()
                   : key.size;
    auto it = pool.blocks.lower_bound(&key);

    // TODO(caikun): will sync device in NPUCachingAllocator
    // npuSynchronizeDevice();

    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin()) return false;
      size_t totalReleased = 0;
      --it;  // Back up one item.  Now on the largest block for the correct
             // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= CachingAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        totalReleased += (*it)->size;
        if (it != pool.blocks.begin()) {
          --it;
          release_block(*cur);
        } else {
          release_block(*cur);
          break;
        }
      }
      if (totalReleased < key.size) {
        return false;
      }
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks(const std::shared_ptr<GatheredContext>& context) {
    // TODO(caikun): will sync device in NPUCachingAllocator
    // npuSynchronizeDevice();

    // First ensure that all blocks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_events(context);

    // Free all non-split cached blocks to system allocator
    release_blocks(large_blocks);
    release_blocks(small_blocks);
    return true;
  }

  void release_expandable_segment(Block* block) {
    TORCH_INTERNAL_ASSERT(block->size == block->expandable_segment_->size(),
                          "block disagrees with segment");
    TORCH_INTERNAL_ASSERT(!block->mapped);
    auto it = std::find(expandable_segments_.begin(),
                        expandable_segments_.end(), block->expandable_segment_);
    TORCH_INTERNAL_ASSERT(it != expandable_segments_.end());
    expandable_segments_.erase(it);
    block->pool->unmapped.erase(block);
    delete block->expandable_segment_;
    block->expandable_segment_ = nullptr;
    delete block;
  }

  void release_block(Block* block) {
    TORCH_INTERNAL_ASSERT(!block->expandable_segment_);
    devproxy::freeDevice(block->ptr);
    total_allocated_memory -= block->size;

    auto* pool = block->pool;
    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(stats.reserved_bytes[stat_type],
                  -static_cast<std::int64_t>(block->size));
    });

    if (block->size >= CachingAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);
    if (record_history) {
      record_trace(TraceEntry::SEGMENT_FREE, int64_t(block->ptr), block->size,
                   block->stream, nullptr);
    }
    pool->blocks.erase(block);
    delete block;
  }

  void unmap_block(Block* block) {
    auto unmapped = block->expandable_segment_->unmap(
        SegmentRange{block->ptr, block->size});
    if (unmapped.size == 0) {
      return;
    }
    block->pool->blocks.erase(block);

    ptrdiff_t before_size =
        static_cast<char*>(unmapped.ptr) - static_cast<char*>(block->ptr);
    if (before_size > 0) {
      // prev? -> before_free -> block
      Block* before_free = new Block(block->device, block->stream, before_size,
                                     block->pool, block->ptr);
      before_free->expandable_segment_ = block->expandable_segment_;
      before_free->splice(block->prev, block);
      block->pool->blocks.insert(before_free);
    }

    auto after_size = block->size - (before_size + unmapped.size);
    if (after_size > 0) {
      // block -> after_free -> next?
      Block* after_free =
          new Block(block->device, block->stream, after_size, block->pool,
                    static_cast<char*>(unmapped.ptr) + unmapped.size);
      after_free->expandable_segment_ = block->expandable_segment_;
      after_free->splice(block, block->next);
      block->pool->blocks.insert(after_free);
    }

    block->ptr = unmapped.ptr;
    block->size = unmapped.size;
    block->mapped = false;

    try_merge_blocks(block, block->prev, *block->pool);
    try_merge_blocks(block, block->next, *block->pool);
    block->pool->unmapped.insert(block);

    // update statistics
    total_allocated_memory -= unmapped.size;
    StatTypes stat_types = get_stat_types_for_pool(*block->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.reserved_bytes[stat_type], -unmapped.size);
    });
    if (record_history) {
      record_trace(TraceEntry::SEGMENT_UNMAP, int64_t(unmapped.ptr),
                   unmapped.size, block->stream, nullptr);
    }
  }
  void release_blocks(BlockPool& pool) {
    std::vector<Block*> to_unmap;
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (block->expandable_segment_) {
        // unmapping will mutate the free pool
        // so just gather what needs to be freed
        // to avoid invalidating the iterator
        to_unmap.push_back(block);
      } else if (!block->prev && !block->next) {
        release_block(block);
      }
    }
    for (Block* block : to_unmap) {
      unmap_block(block);
      if (!block->prev && !block->next) {
        release_expandable_segment(block);
      }
    }
  }

  void synchronize_and_free_events(
      const std::shared_ptr<GatheredContext>& context) {
    // Synchronize on outstanding events and then free associated blocks.
    for (auto& st : events) {
      for (auto& e : st.second) {
        DIPUEvent event = std::move(e.first);
        Block* block = e.second;
        event.synchronize();

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
      }
    }

    events.clear();
  }

  // TODO(caikun): implement diff from NPUCachingAllocator
  void insert_events(Block* block) {
    devapis::deviceId_t prev_device = devproxy::current_device();

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      devproxy::setDevice(stream.device_index());

      DIPUEvent event;
      event.record(stream);

      block->event_count++;
      events[stream].emplace_back(std::move(event), block);
    }

    devproxy::setDevice(prev_device);
  }

  void process_events(const std::shared_ptr<GatheredContext>& context) {
    // Process outstanding cudaEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = events.begin(); it != events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        DIPUEvent event = std::move(e.first);
        Block* block = e.second;

        if (!event.query()) {
          // Return the ownership of the Event (unique ptr)
          e.first = std::move(event);
          break;
        }

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block, context);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Iterates over sizes of all memory blocks for given device in given pool
  static void cache_info_aux(const BlockPool& pool, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

  // TODO(caikun): no device in TraceEntry, have device in NPUCachingAllocator.
  void record_trace(TraceEntry::Action action, int64_t addr, size_t size,
                    deviceStream_t stream,
                    std::shared_ptr<GatheredContext> context) {
    auto te = TraceEntry(
        action, addr, size, stream,
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);
    if (alloc_trace->size() < alloc_trace_max_entries_) {
      alloc_trace->emplace_back(te);
    } else {
      (*alloc_trace)[alloc_trace_next++] = te;
      if (alloc_trace_next == alloc_trace_max_entries_) {
        alloc_trace_next = 0;
      }
    }
  }
};  // namespace

void local_raw_delete(void* ptr);

class NativeCachingAllocator : public DeviceAllocator {
 private:
  std::mutex mutex;

  // allocated blocks by device pointer
  ska::flat_hash_map<void*, Block*> allocated_blocks;

  void add_allocated_block(Block* block) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_blocks[block->ptr] = block;
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;
  std::vector<std::unique_ptr<AllocatorMetrics>> metrics_producer;

  Block* get_allocated_block(void* ptr, bool remove = false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_blocks.find(ptr);
    if (it == allocated_blocks.end()) {
      return nullptr;
    }
    Block* block = it->second;
    if (remove) {
      allocated_blocks.erase(it);
    }
    return block;
  }

  void init(int device_count) override {
    const auto size = static_cast<int64_t>(device_allocator.size());
    if (size < device_count) {
      device_allocator.resize(device_count);
      metrics_producer.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
        static const metrics::Collector::labelset lable_set{
            {"type", "caching"}};
        metrics_producer[i] = std::make_unique<AllocatorMetrics>(lable_set);
        metrics_producer[i]->set_device_number(std::to_string(i));
      }
    }
  }

  bool initialized() override { return !device_allocator.empty(); }

  /** allocates a block which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, deviceStream_t stream) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ", device,
        ": did you call init?");
    Block* block = device_allocator[device]->malloc(device, size, stream);
    add_allocated_block(block);
    *devPtr = block->ptr;
    metrics_producer[device]->allocate(block->ptr, block->size);
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Block* block = get_allocated_block(ptr, true /* remove */);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    metrics_producer[block->device]->deallocate(block->ptr);
    device_allocator[block->device]->free(block);
  }

  void setMemoryFraction(double fraction, int device) override {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ", device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(0 <= fraction && fraction <= 1,
                          "invalid fraction:", fraction,
                          ". Please set within (0, 1).");
    devproxy::setDevice(static_cast<c10::DeviceIndex>(device));
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries,
                     RecordContext when) override {
    for (auto& allocator : device_allocator) {
      allocator->recordHistory(enabled, context_recorder,
                               alloc_trace_max_entries, when);
    }
  }

  bool isHistoryEnabled() override {
    int device = devproxy::current_device();
    return device_allocator[device]->isHistoryEnabled();
  }

  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    for (auto& allocator : device_allocator) {
      allocator->attachOutOfMemoryObserver(std::move(observer));
    }
  }

  void emptyCache() override {
    for (auto& da : device_allocator) {
      da->emptyCache();
    }
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) override {
    Block* block = get_allocated_block(ptr);
    if (!block) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return device_allocator[block->device]->getBaseAllocation(block, outSize);
  }

  void recordStream(const c10::DataPtr& ptr,
                    const DIPUStream& stream) override {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when CUDA tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &local_raw_delete) {
      return;
    }

    Block* block = get_allocated_block(ptr.get());
    // block must not be null reaching here
    TORCH_INTERNAL_ASSERT(block != nullptr, "No allocated block can be found");
    device_allocator[block->device]->recordStream(block, stream);
  }

  SnapshotInfo snapshot() override {
    SnapshotInfo result;
    for (auto& da : device_allocator) {
      result.device_traces.emplace_back(da->trace());
      auto snap = da->snapshot();
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }
    return result;
  }

  c10::DataPtr allocate(size_t size) const override {
    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError, size < one_exa_bytes,
        "CUDA out of memory. Tried to allocate more than 1EB memory.");
    devapis::deviceId_t device = devproxy::current_device();
    void* r = nullptr;
    if (size != 0) {
      // Allocator declars allocate const!?
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<NativeCachingAllocator*>(this)->malloc(
          &r, device, size,
          dipu::getCurrentDIPUStream(static_cast<c10::DeviceIndex>(device))
              .rawstream());
    }
    return {r, r, &local_raw_delete,
            c10::Device(dipu::DIPU_DEVICE_TYPE,
                        static_cast<c10::DeviceIndex>(device))};
  }

  c10::DeleterFnPtr raw_deleter() const override { return &local_raw_delete; }

  void cacheInfo(int dev_id, size_t* largestBlock) override {
    device_allocator[dev_id]->cacheInfo(largestBlock);
  }

  void assertValidDevice(int device) const {
    const auto device_num = device_allocator.size();
    TORCH_CHECK(0 <= device && device < static_cast<int64_t>(device_num),
                "Invalid device argument ", device, ": did you call init?");
  }

  DeviceStats getDeviceStats(int device) override {
    assertValidDevice(device);
    return device_allocator[device]->getStats();
  }

  void resetAccumulatedStats(int device) override {
    assertValidDevice(device);
    device_allocator[device]->resetAccumulatedStats();
  }

  void resetPeakStats(int device) override {
    assertValidDevice(device);
    device_allocator[device]->resetPeakStats();
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device = devproxy::current_device();
    void* r = nullptr;
    malloc(&r, device, nbytes,
           dipu::getCurrentDIPUStream(static_cast<c10::DeviceIndex>(device))
               .rawstream());
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, deviceStream_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device = devproxy::current_device();
    void* r = nullptr;
    malloc(&r, device, nbytes, stream);
    return r;
  }

  void raw_delete(void* ptr) override { this->free(ptr); }

  std::string name() override { return "torch"; }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
NativeCachingAllocator* native_caching_allocator = new NativeCachingAllocator();
REGISTER_ALLOCATOR(dipu::DIPU_DEVICE_TYPE, native_caching_allocator);

void local_raw_delete(void* ptr) { native_caching_allocator->free(ptr); }

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  if (!isTorchAllocator()) {
    DIPU_LOGW("Not using torch allocator, skipping setAllocatorSettings");
    return;
  }
  CachingAllocatorConfig::instance().parseArgs(env.c_str());
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
// Size pretty-printer
std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (static_cast<double>(size) / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << (static_cast<double>(size) / 1048576.0);
    os << " MiB";
  } else {
    os << (static_cast<double>(size) / 1073741824.0);
    os << " GiB";
  }
  return os.str();
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

struct BackendStaticInitializer {
  BackendStaticInitializer() {
    allocator.store(native_caching_allocator);
    init(devproxy::getDeviceCount());
  }
};

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<DeviceAllocator*> allocator;
BackendStaticInitializer backend_static_initializer;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

}  // namespace dipu::allocator
// NOLINTEND
