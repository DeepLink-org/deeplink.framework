// Copyright (c) 2023, DeepLink.

#include <execinfo.h>
#include <cxxabi.h>

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <c10/util/Exception.h>

#include "DIPUCachingAllocator.h"
#include "DIPUGuard.h"
#include "DIPUEvent.h"

using dipu::DIPUEvent;

#if USE_PROFILE
#include <c10/util/ThreadLocalDebugInfo.h>
#endif

#define MASK_WORDS 1

static bool native_memory_strategy = false;

namespace dipu {

class BoundException : public std::exception {
  const char * what() const throw() {
    return "dipu memory out of bounds!";
  }
};

class ManageException : public std::exception {
  const char * what() const throw() {
    return "dipu memory out of allocator!";
  }
};

// the constant parameters for chunk
static constexpr size_t minimum_round_size =
    512;  // all chunks are rounded at least 512 bytes
static constexpr size_t small_allocation_size =
    1048576;  // maximum for "small" allocation is 1 Mib
static constexpr size_t small_buffer_size =
    2097152;  // "small" allocations are in 2 Mibs chunks
static constexpr size_t large_allocation_size =
    10485760;  // allocation sizes from 1 Mib to 10 Mibs use larger chunks
static constexpr size_t large_buffer_size =
    20971520;  // "large" allocations may be in 20 Mibs chunks
static constexpr size_t maximum_round_size =
    2097152;  // all chunks are rounded at most 2 Mibs

// DEBUG_MODE: the mask size
static constexpr size_t mask_bytes = MASK_WORDS * sizeof(int64_t);

// DEBUG_MODE: Debugging Flag
static bool debug_mode = false;

// DEBUG_MODE: backtrace layers
static constexpr int layer_num = 10;

static std::shared_ptr<int64_t> newMask(int64_t magic_word) {
  std::shared_ptr<int64_t> m (new int64_t[MASK_WORDS], std::default_delete<int64_t[]>());
  for (int i = 0; i < MASK_WORDS; ++i) {
    m.get()[i] = magic_word;
  }
  return m;
}

static std::shared_ptr<int64_t> header_mask = newMask(0x4c4955595558494e);
static std::shared_ptr<int64_t> footer_mask = newMask(0x48574a4341544348);

class MemoryStats {
public:
  uint64_t   allocated_size;      // total size allocated in bytes
  uint64_t   max_allocated_size;  // max total size allocated in bytes
  uint64_t   cached_size;         // total size in cache in bytes
  uint64_t   max_cached_size;     // max total size in cache in bytes

public:
  MemoryStats() :
      allocated_size(0), max_allocated_size(0),
      cached_size(0), max_cached_size(0) { }

  virtual void allocated(size_t num_allocated) {
    allocated_size += num_allocated;
    max_allocated_size = std::max(max_allocated_size, allocated_size);
  }

  virtual void deallocated(size_t num_allocated) {
    allocated_size -= num_allocated;
  }

  virtual void cached(size_t num_cached) {
    cached_size += num_cached;
    max_cached_size = std::max(max_cached_size, cached_size);
  }

  virtual void decached(size_t num_cached) {
    cached_size -= num_cached;
  }
};
// ChunkPool is a sorted list of Chunk, using pointer for comparing
struct Chunk;
typedef bool (*Comparison)(const Chunk*, const Chunk*);
typedef std::set<Chunk*, Comparison> ChunkPool;
using stream_set = std::unordered_set<dipu::DIPUStream>;

struct Chunk {
  int device_id;      // dipu device id
  deviceStream_t stream;  // allocation stream
  stream_set streams_inuse;  // streams on which the chunk was used
  size_t size;         // chunk size in bytes
  ChunkPool* pool;     // owning memory pool
  void* ptr;           // memory address
  bool allocated;      // is_allocated flag
  Chunk* prev;         // prev chunk if split from a larger allocation
  Chunk* next;         // next chunk if split from a larger allocation
  int event_count;  // number of outstanding DIPU events.
  Chunk(int device_id, deviceStream_t stream, size_t size, ChunkPool* pool,
        void* ptr)
      : device_id(device_id),
        stream(stream),
        size(size),
        pool(pool),
        ptr(ptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        streams_inuse(),
        event_count(0) {}

  // constructor for search key
  Chunk(int device_id, deviceStream_t stream, size_t size)
      : device_id(device_id),
        stream(stream),
        size(size),
        pool(nullptr),
        ptr(nullptr),
        allocated(0),
        prev(nullptr),
        next(nullptr),
        streams_inuse(),
        event_count(0) {}
};

static bool ChunkComparator(const Chunk* a, const Chunk* b) {
  if (a->device_id != b->device_id) {
    return a->device_id < b->device_id;
  }
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

// format size(byte) in string
static std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

class SimpleCachingAllocImpl {
protected:
  // Memory statistics
  std::vector<MemoryStats> memory_stats;

  // lock around all operations
  std::recursive_mutex base_mutex;

  // lock around calls to free
  std::mutex dipu_mutex;

  // cached chunks are larger than 1 MB
  ChunkPool large_chunks;

  // cached chunks are 1 MB or smaller
  ChunkPool small_chunks;

  // allocated chunks by device pointer
  std::unordered_map<void*, Chunk*> allocated_chunks;

  // outstanding dipu events
  std::deque<std::pair<std::shared_ptr<DIPUEvent>, Chunk*>> events_;

  int count = 0;

public:
  SimpleCachingAllocImpl()
      :large_chunks(ChunkComparator), small_chunks(ChunkComparator) {
    count = 0;
  }

  ~SimpleCachingAllocImpl() {
  }

  void malloc(void** data_ptr, size_t size, deviceStream_t stream, int device_id) {
    std::lock_guard<std::recursive_mutex> lock(base_mutex);

    process_events();

    size = roundUpSize(size);

    auto &stats = get_memory_stats_for_device(device_id);

    Chunk searchChunk(device_id, stream, size);
    auto& pool = getChunkPool(size);

    auto findFreeChunk = [&]() -> Chunk* {
      auto it = pool.lower_bound(&searchChunk);
      if (it != pool.end() && (*it)->device_id == device_id &&
          (*it)->stream == stream) {
        Chunk* chunk = *it;
        pool.erase(it);
        return chunk;
      }
      return nullptr;
    };

    Chunk* chunk = findFreeChunk();
    if (chunk == nullptr) {
      void* ptr;
      size_t allocation_size = getAllocationSize(size);
      dipuMalloc(device_id, &ptr, allocation_size, stream);
      stats.cached(allocation_size);
      chunk = new Chunk(device_id, stream, allocation_size, &pool, ptr);
      carveHeader(chunk);
      carveFooter(chunk);
    }

    Chunk* remain_chunk = nullptr;
    if (shouldSplit(chunk, size)) {
      remain_chunk = chunk;

      chunk = new Chunk(device_id, stream, size, &pool, chunk->ptr);
      chunk->prev = remain_chunk->prev;
      if (chunk->prev) {
        chunk->prev->next = chunk;
      }
      chunk->next = remain_chunk;

      remain_chunk->prev = chunk;
      remain_chunk->ptr = static_cast<char*>(remain_chunk->ptr) + size;
      remain_chunk->size -= size;
      carveMasks(chunk, remain_chunk);
      pool.insert(remain_chunk);
    }

    chunk->allocated = true;
    allocated_chunks[chunk->ptr] = chunk;

    *data_ptr = chunk->ptr;

    stats.allocated(chunk->size);

    recordBacktrace(chunk);

    #if USE_PROFILE
    bool profile_memory = c10::memoryProfilingEnabled();
    if (profile_memory) {
      c10::reportMemoryUsageToProfiler(
      chunk, chunk->size, c10::Device(dipu::DIPU_DEVICE_TYPE, device_id));
    }
    #endif
  }

  void free(void* ptr) {
    std::lock_guard<std::recursive_mutex> lock(base_mutex);
    if (!ptr) {
      return;
    }

    auto it = allocated_chunks.find(ptr);
    if (it == allocated_chunks.end()) {
      AT_ERROR("invalid device pointer: ", ptr);
    }

    Chunk* chunk = it->second;
    allocated_chunks.erase(it);
    chunk->allocated = false;

    get_memory_stats_for_device(chunk->device_id).deallocated(chunk->size);

    #if USE_PROFILE

    bool profile_memory = c10::memoryProfilingEnabled();
    if (profile_memory) {
    c10::reportMemoryUsageToProfiler(
        chunk, -chunk->size, c10::Device(c10::DeviceType::CUDA, chunk->device_id));
    }
    #endif

    if (!chunk->streams_inuse.empty()) {
      record_event(chunk);
    } else {
      giveBackChunk(chunk);
    }
  }

  void recordStream(const c10::DataPtr& data_ptr, dipu::DIPUStream stream) {
    auto ptr = data_ptr.get();
    if (!ptr) {
      return;
    }

    std::lock_guard<std::recursive_mutex> lock(std::mutex);

    if (debug_mode) {
      ptr = static_cast<char*>(ptr) - mask_bytes;
    }
    auto it = allocated_chunks.find(ptr);
    if (it == allocated_chunks.end()) {
      AT_ERROR("invalid device pointer, No allocated chunk can be found: ", ptr);
    }

    Chunk* chunk = it->second;
    if (stream.rawstream() == chunk->stream) {
        return;
    }
    chunk->streams_inuse.insert(stream);
  }

  void emptyCached() {
    std::lock_guard<std::recursive_mutex> lock(base_mutex);
    synchronize_and_free_events();
    freeChunks(large_chunks, large_chunks.begin(), large_chunks.end());
    freeChunks(small_chunks, small_chunks.begin(), small_chunks.end());
  }

  virtual MemoryStats& get_memory_stats_for_device(int device) {
    auto dev_count = devproxy::getDeviceCount();
    auto cur_device = devproxy::current_device();
    device = device == -1 ? cur_device : device;
    if (device >=0 && device < dev_count) {
      if ((size_t) device >= memory_stats.size()) {
        memory_stats.resize(device + 1);
      }
      return memory_stats.at(device);
    } else {
      LOG(FATAL) << "Caching Allocator: wrong device!";
      TORCH_CHECK(false, "get_memory_stats_for_device: device idx error");
    }
  }

protected:
  void process_events() {
    while (!events_.empty()) {
      auto& n = events_.front();
      auto event_ptr = n.first;
      Chunk* chunk = n.second;
      dipu::DIPUGuard guard(event_ptr->device_index());
      const bool ret = event_ptr->query();
      if (ret == false) {
        break;
      }
      chunk->event_count--;
      if (chunk->event_count == 0) {
        giveBackChunk(chunk);
      }
      events_.pop_front();
    }
  }

  void record_event(Chunk* chunk) {
    stream_set stream_inuse(std::move(chunk->streams_inuse));
    AT_ASSERT(chunk->streams_inuse.empty());
    for (auto it = stream_inuse.begin(); it != stream_inuse.end(); ++it) {
      c10::DeviceIndex device_id = static_cast<c10::DeviceIndex>(it->device_index());
      /// enhance to use event cache in future
      std::shared_ptr<DIPUEvent> event_ptr = std::make_shared<DIPUEvent>();
      event_ptr->record(*it);
      chunk->event_count++;
      events_.emplace_back(event_ptr, chunk);
    }
  }

  void synchronize_and_free_events() {
    for (auto& n : events_) {
      auto event_ptr = n.first;
      Chunk* chunk = n.second;
      event_ptr->synchronize();
       /// enhance to return to event cache
      chunk->event_count--;
      if (chunk->event_count == 0) {
        giveBackChunk(chunk);
      }
      events_.pop_front();
    }
  }

//virtual 
protected:
  virtual void carveMasks(Chunk* chunk, Chunk* remain_chunk) {}

  virtual void carveHeader(Chunk* chunk) {}

  virtual void carveFooter(Chunk* chunk) {}

  virtual bool checkMask(Chunk* chunk) { return true;}

  virtual void recordBacktrace(Chunk* chunk) {}

  virtual void dipuMalloc(int device, void** data_ptr, size_t size, deviceStream_t& stream) {
    // first using malloc, if fails then free all cached chunks and remalloc
    auto status = devproxy::mallocDevice(data_ptr, size, false);
    if (status != devapis::OpStatus::SUCCESS) {
      free_cached_chunk(device);
      devproxy::mallocDevice(data_ptr, size);
      // *********** wait devapi to add device status  ******//
    }
  }

  virtual size_t roundUpSize(size_t size) {
    if (size < minimum_round_size) {
      return minimum_round_size;
    } else {
      return minimum_round_size *
             ((size + minimum_round_size - 1) / minimum_round_size);
    }
  }

// not virtual
protected:
  // moves a chunk into a pool of cached free chunks
  void giveBackChunk(Chunk* chunk) {
    checkMask(chunk);
    AT_ASSERT(!chunk->allocated && chunk->event_count == 0);
    auto& pool = *chunk->pool;
    mergeChunks(chunk, chunk->prev, pool);
    mergeChunks(chunk, chunk->next, pool);
    pool.insert(chunk);
  }

  // combine previously split chunks
  void mergeChunks(Chunk* dst, Chunk* src, ChunkPool& pool) {
    if (!src || src->allocated || src->event_count > 0) {
      return;
    }
    if (dst->prev == src) {
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else {
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    dst->size += src->size;
    pool.erase(src);
    delete src;
  }

  // get chunk pool
  ChunkPool& getChunkPool(size_t size) {
    if (size <= small_allocation_size) {
      return small_chunks;
    } else {
      return large_chunks;
    }
  }

  // get allocation size
  size_t getAllocationSize(size_t size) {
    auto native_memory_strategy = get_memory_strategy();
    size_t malloc_size = size;
    if (size <= small_allocation_size) {
      return small_buffer_size;
    } else {
      if (malloc_size < large_allocation_size) {
        return large_buffer_size;
      } else {
        return maximum_round_size *
               ((malloc_size + maximum_round_size - 1) / maximum_round_size);
      }
    }
  }

  bool shouldSplit(Chunk* chunk, size_t size) {
    size_t remaining = chunk->size - size;
    if (chunk->pool == &small_chunks) {
      return remaining >= minimum_round_size;
    } else if (chunk->pool == &large_chunks) {
      return remaining > small_allocation_size;
    } else {
      AT_ERROR("shouldSplit: invalid DIPU chunk pool");
    }
  }

  void free_cached_chunk(int device) {
    synchronize_and_free_events();

    Chunk lower_bound(device, nullptr, 0);
    Chunk upper_bound(device + 1, nullptr, 0);

    freeChunks(large_chunks, large_chunks.lower_bound(&lower_bound),
               large_chunks.lower_bound(&upper_bound));
    freeChunks(small_chunks, small_chunks.lower_bound(&lower_bound),
               small_chunks.lower_bound(&upper_bound));
  }

  void freeChunks(ChunkPool& chunks, ChunkPool::iterator it,
                  ChunkPool::iterator end) {
    std::lock_guard<std::mutex> lock(dipu_mutex);
    while (it != end) {
      Chunk* chunk = *it;
      if (!chunk->prev && !chunk->next) {
        devproxy::freeDevice((void*)chunk->ptr);
        get_memory_stats_for_device(chunk->device_id).decached(chunk->size);
        auto cur = it;
        ++it;
        chunks.erase(cur);
        delete chunk;
      } else {
        ++it;
      }
    }
  }
};

/// remove DebugAllocator which copy from camb, very stupid design, redesign in future


// *** change to use base class ptr in future
// allocator using for memory management
auto realAllocator = std::make_unique<SimpleCachingAllocImpl>();

// The library provides a recordStream() function to help insert the correct
// synchronization when allocations are used on multiple streams. This will
// ensure that the chunk is not reused before each recorded stream completes
// work.
void recordStream(const c10::DataPtr& data_ptr, dipu::DIPUStream stream) {
  realAllocator->recordStream(data_ptr, stream);
}

// ** change to call when create real allocator instance
inline void retriveDebugFlag() {
  char* env = std::getenv("ENABLE_CATCH_MEMORY_DEBUG");
  if (env != NULL) {
    debug_mode = (*env == '1');
  } else {
    debug_mode = false;
  }
}

static void DIPUCachingDeleter(void* ptr) {
  if (ptr == nullptr) return;
  realAllocator->free(ptr);
}

c10::DataPtr DIPUCachingAllocator::allocate(size_t size) const {
  // fake allocation, only for setting device
  auto device_id = devproxy::current_device();
  return allocate(size, devproxy::current_device());
}

c10::DataPtr DIPUCachingAllocator::allocate(size_t size,
                                           c10::DeviceIndex device_id) const {
  void* data = nullptr;
  if (size <= 0) {
    return {data, data, &DIPUCachingDeleter,
      c10::Device(dipu::DIPU_DEVICE_TYPE, device_id)};
  }
  // if (debug_mode) {
  //   debugging_allocator.malloc(&data, size, dipu::getCurrentDIPUStream(device_id).rawstream(),
  //       static_cast<int>(device_id));
  //   data = static_cast<char*>(data) + mask_bytes;

  realAllocator->malloc(&data, size, dipu::getCurrentDIPUStream(device_id).rawstream(),
      static_cast<int>(device_id));
  return {data, data, &DIPUCachingDeleter,
          c10::Device(dipu::DIPU_DEVICE_TYPE, device_id)};
}

c10::DeleterFnPtr DIPUCachingAllocator::raw_deleter() const {
  return &DIPUCachingDeleter;
}

void set_memory_strategy(bool ms) {
  native_memory_strategy = ms;
}

bool get_memory_strategy() { return native_memory_strategy; }

// return the current memory allocated on dipu
uint64_t currentMemoryAllocated(int device_id) {
  return realAllocator->get_memory_stats_for_device(device_id).allocated_size;
  
}

// return the current memory cached on dipu
uint64_t currentMemoryCached(int device_id) {
  return realAllocator->get_memory_stats_for_device(device_id).cached_size;

}

// return the max memory allocated on dipu
uint64_t maxMemoryAllocated(int device_id) {
  return realAllocator->get_memory_stats_for_device(device_id).max_allocated_size;
}

// return the max memory cached on dipu
uint64_t maxMemoryCached(int device_id) {
  return realAllocator->get_memory_stats_for_device(device_id).max_cached_size;
}

// empty all cached and unchained memory
void emptyCachedMem() {
  return realAllocator->emptyCached();
}


}  // namespace dipu