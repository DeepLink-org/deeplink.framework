// Copyright (c) 2023, DeepLink.

#include "DIPUCachingHostAllocator.h"

#include <stdint.h>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <c10/util/Exception.h>

#include <csrc_dipu/runtime/device/deviceapis.h>
#include "DIPUGuard.h"
#include "DIPUEvent.h"

namespace dipu {

namespace {

struct BlockSize {
  size_t size_{0};
  void* ptr_{nullptr};
};

struct Block {
  size_t size_{0};
  void* ptr_{nullptr};

  std::mutex mutex_;
  bool allocated_{false};
  size_t event_count_{0};
  std::unordered_set<DIPUStream> streams_;
};

// Note: dipu::devapis::createEvent when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling dipu::devapis::createEvent/destroyEvent. This results in
// significant improvements in multithreaded workloads with high allocation rates.
class EventPool {
 public:
  using Event = std::unique_ptr<DIPUEvent, std::function<void(DIPUEvent*)>>;
  EventPool() : pools_(devapis::getDeviceCount()) {}

  Event get(c10::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<c10::DeviceIndex>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](DIPUEvent* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<DIPUEvent>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on destruction.
    return Event(
        new DIPUEvent(),
        destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<DIPUEvent>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// Used for heterogenous lookup support in the free list.
struct BlockComparator {
  using is_transparent = void;
  bool operator()(const Block* a, const Block* b) const {
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b->ptr_;
  }

  // Transparent overloads
  bool operator()(const Block* a, BlockSize b) const {
    if (a->size_ != b.size_) {
      return a->size_ < b.size_;
    }
    return (uintptr_t)a->ptr_ < (uintptr_t)b.ptr_;
  }
  bool operator()(BlockSize a, const Block* b) const {
    if (a.size_ != b->size_) {
      return a.size_ < b->size_;
    }
    return (uintptr_t)a.ptr_ < (uintptr_t)b->ptr_;
  }
};

/**
 * Note [DIPUHostAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * We have three key data structures - the free list which stores blocks that
 * are not currently used, the block list which stores all blocks that have been
 * allocated, and the event queue which stores DIPU events and their
 * corresponding blocks.
 *
 * Each of these are protected by a separate mutex. The key design principles
 * are to 1) only hold each mutex for the minimal amount of time possible, 2)
 * never do any possible expensive operations (such as CUDA runtime API calls)
 * while holding the lock.
 *
 * There are three public methods: allocate, free, and record_event. In the
 * allocate path, we first check to see if we can service our request from this
 * free list, and otherwise we create a new block with dipu::devapis::mallocHost.
 * In the free path, we insert events (if required) into the event queue, and if
 * possible insert our block back into the free list. In allocate, we first
 * eagerly query events until we find one that is not ready, and insert the
 * corresponding block onto the free list if all the events recorded for a
 * block are ready. In the record_event path, we simply insert the given
 * stream into the set of streams tracked by the specified block. This set of
 * streams is then consumed in the free path.
 *
 * Some of the invariants here are less strict than they could be - for example,
 * we do not enforce that free(Block* block) => block->event_count == 0. This is
 * for compatibility reasons, and we can explore enforcing these in subsequent
 * versions.
 */

class DIPUHostAllocator {
 public:
  std::pair<void*, void*> allocate(size_t size) {
    if (size == 0) {
      return {nullptr, nullptr};
    }

    process_events();

    // First, try to allocate from the free list
    {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      auto it = free_list_.lower_bound(BlockSize{size, nullptr});
      if (it != free_list_.end()) {
        auto block = *it;
        block->allocated_ = true;
        free_list_.erase(it);
        return {block->ptr_, reinterpret_cast<void*>(block)};
      }
    }

    // Then, create a new block.
    // Round up the allocation to the nearest power of two to improve reuse.
    void* ptr = nullptr;
    uint64_t allocate_size = c10::llvm::PowerOf2Ceil(size);
    devapis::mallocHost(&ptr, allocate_size);
    auto block = new Block();
    block->size_ = allocate_size;
    block->ptr_ = ptr;
    block->allocated_ = true;

    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      blocks_.insert(block);
      ptr_to_block_.insert({block->ptr_, block});
    }
    return {block->ptr_, reinterpret_cast<void*>(block)};
  }

  void free(void* ctx) {
    if (!ctx) {
      return;
    }

    // Note: we can assume that free is correctly paired with alloc,
    // and thus we do not need to look up the ctx in blocks_.
    auto* block = reinterpret_cast<Block*>(ctx);

    c10::optional<std::vector<EventPool::Event>> events;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<EventPool::Event>();
        events->reserve(block->streams_.size());
        for (auto stream : block->streams_) {
          auto event = event_pool_.get(stream.device_index());
          event->record(stream);
          events->push_back(std::move(event));
        }
        block->event_count_ += events->size();
        block->streams_.clear();
      }
    }

    if (!events) {
      std::lock_guard<std::mutex> g(free_list_mutex_);
      free_list_.insert(block);
    } else {
      std::lock_guard<std::mutex> g(dipu_events_mutex_);
      for (auto&& event : *events) {
        dipu_events_.emplace_front(std::move(event), block);
      }
    }
  }

  bool record_event(void* ptr, void* ctx, DIPUStream stream) {
    auto* block = reinterpret_cast<Block*>(ctx);

    // Note: we need to check if the passed-in `ctx` is valid. This is because
    // `record_event` (via `CachingHostAllocator_recordEvent`) can be invoked on
    // an arbitrary tensor, and is not guaranteed to correspond to a pinned
    // memory allocation. Therefore, we need to check that `ctx` is valid before
    // proceeding.
    {
      std::lock_guard<std::mutex> g(blocks_mutex_);
      if (blocks_.find(block) != blocks_.end()) {
        // Now we know this object is safe to access.
        std::lock_guard<std::mutex> gb(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }

      auto it = ptr_to_block_.find(ptr);
      if (it != ptr_to_block_.end()) {
        block = it->second;
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(block->allocated_);
        block->streams_.insert(stream);
        return true;
      }
    }

    return false;
  }

  void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

    // Release cached events from the event pool.
    event_pool_.empty_cache();

    // Remove all elements from the free list, remove them from the blocks
    // list, and free the associated pinned memory allocation. This requires
    // concurrently holding both the free list mutex and the blocks mutex, and
    // is the only function that concurrently holds multiple mutexes.
    std::lock(free_list_mutex_, blocks_mutex_);
    std::lock_guard<std::mutex> gf(free_list_mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> gb(blocks_mutex_, std::adopt_lock);

    std::vector<Block*> blocks_to_remove(free_list_.begin(), free_list_.end());
    free_list_.clear();
    for (auto* block : blocks_to_remove) {
      blocks_.erase(block);
      ptr_to_block_.erase(block->ptr_);
      devapis::freeHost(block->ptr_);
      delete block;
    }
  }

private:
  void process_events() {
    while (true) {
      // Avoid calling destroyEvent while holding a mutex, so move
      // intermediate events out of the lock into this object.
      c10::optional<std::pair<EventPool::Event, Block*>> processed;
      {
        std::lock_guard<std::mutex> g(dipu_events_mutex_);
        if (!dipu_events_.empty()) {
          processed = std::move(dipu_events_.back());
          dipu_events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        devapis::EventStatus status = devapis::getEventStatus(*event);
        if (status == devapis::EventStatus::PENDING) {
          // push the event onto the back of the queue if it's not ready.
          {
            std::lock_guard<std::mutex> g(dipu_events_mutex_);
            dipu_events_.push_back(std::move(*processed));
          }
          return;
        } else {
          TORCH_CHECK(status == devapis::EventStatus::READY, "getEventStatus error, status = ", static_cast<int32_t>(status));
        }
      }

      // Process the events.
      TORCH_INTERNAL_ASSERT(processed);
      auto* block = processed->second;
      bool available = false;
      {
        std::lock_guard<std::mutex> g(block->mutex_);
        TORCH_INTERNAL_ASSERT(!block->allocated_)
        block->event_count_--;
        if (block->event_count_ == 0) {
          available = true;
        }
      }

      if (available) {
        std::lock_guard<std::mutex> g(free_list_mutex_);
        free_list_.insert(block);
      }
    }
  }

private:
  EventPool event_pool_;

  alignas(64) std::mutex blocks_mutex_;
  std::unordered_set<Block*> blocks_;
  std::unordered_map<void*, Block*> ptr_to_block_;
  // Note: sharding this mutex seems to be profitable in heavily multi-threaded scenarios.
  alignas(64) std::mutex free_list_mutex_;
  // Note: an alternative datastructure can yield significant wins here in microbenchmarks.
  std::set<Block*, BlockComparator> free_list_;

  alignas(64) std::mutex dipu_events_mutex_;
  std::deque<std::pair<EventPool::Event, Block*>> dipu_events_;
};

}  // anonymous namespace

static DIPUHostAllocator& getDIPUHostAllocator() {
  // leak and don't worry about shutdown
  static auto* r = new DIPUHostAllocator();
  return *r;
}

static void DIPUHostAllocatorDeleter(void* ctx) {
  getDIPUHostAllocator().free(ctx);
}

bool CachingHostAllocator_recordEvent(void* ptr, void* ctx, DIPUStream stream) {
  return getDIPUHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via dipu::devapis::freeHost
void CachingHostAllocator_emptyCache() {
  getDIPUHostAllocator().empty_cache();
}

struct DIPUHostAllocatorWrapper final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    auto ptr_and_ctx = getDIPUHostAllocator().allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &DIPUHostAllocatorDeleter,
        at::DeviceType::CPU};
  }
};

static DIPUHostAllocatorWrapper dipu_host_allocator;

at::Allocator* getCachingHostAllocator() {
  return &dipu_host_allocator;
}

}  // namespace dipu
