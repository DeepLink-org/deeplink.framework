// Copyright (c) 2024, DeepLink.
#include "csrc_dipu/runtime/core/allocator/DIPUCachingHostAllocator.h"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "csrc_dipu/runtime/core/DIPUEvent.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu::allocator {
namespace {

constexpr int32_t kAlignPack(64);

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

// Used for heterogenous lookup support in the free list.
struct BlockComparator {
  using is_transparent = void;
  bool operator()(const Block* a, const Block* b) const {
    if (a->size_ != b->size_) {
      return a->size_ < b->size_;
    }
    return reinterpret_cast<uintptr_t>(a->ptr_) <
           reinterpret_cast<uintptr_t>(b->ptr_);
  }

  // Transparent overloads
  bool operator()(const Block* a, BlockSize b) const {
    if (a->size_ != b.size_) {
      return a->size_ < b.size_;
    }
    return reinterpret_cast<uintptr_t>(a->ptr_) <
           reinterpret_cast<uintptr_t>(b.ptr_);
  }
  bool operator()(BlockSize a, const Block* b) const {
    if (a.size_ != b->size_) {
      return a.size_ < b->size_;
    }
    return reinterpret_cast<uintptr_t>(a.ptr_) <
           reinterpret_cast<uintptr_t>(b->ptr_);
  }
};

/**
 * Note [DIPUCachingHostAllocator design]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * We have three key data structures - the free list which stores blocks that
 * are not currently used, the block list which stores all blocks that have been
 * allocated, and the event queue which stores events and their
 * corresponding blocks.
 *
 * Each of these are protected by a separate mutex. The key design principles
 * are to 1) only hold each mutex for the minimal amount of time possible, 2)
 * never do any possible expensive operations (such as CUDA runtime API calls)
 * while holding the lock.
 *
 * There are three public methods: allocate, free, and record_event. In the
 * allocate path, we first check to see if we can service our request from this
 * free list, and otherwise we create a new block with mallocHost. In the
 * free path, we insert events (if required) into the event queue, and if
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
class DIPUCachingHostAllocator {
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

    // Round up the allocation to the nearest power of two to improve reuse.
    void* ptr = nullptr;
    size_t allocation_size = c10::llvm::PowerOf2Ceil(size);
    devproxy::mallocHost(&ptr, allocation_size);
    auto block = new Block();
    block->size_ = allocation_size;
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

    c10::optional<std::vector<DIPUEvent>> events;
    {
      std::lock_guard<std::mutex> g(block->mutex_);
      block->allocated_ = false;
      if (block->streams_.empty()) {
        TORCH_INTERNAL_ASSERT(block->event_count_ == 0);
      } else {
        events = std::vector<DIPUEvent>();
        events->reserve(block->streams_.size());
        for (auto stream : block->streams_) {
          DIPUEvent event;
          event.record(stream);
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
      std::lock_guard<std::mutex> g(events_mutex_);
      for (auto&& event : *events) {
        events_.emplace_front(std::move(event), block);
      }
    }
  }

  bool record_event(void* ptr, void* ctx, const DIPUStream& stream) {
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

    // We need do some sync like syncStream when record_event return false.
    // It will happen when copy from host to device and host tensor allocated by
    // malloc.
    return false;
  }

  void empty_cache() {
    // Flush any available blocks into the free_list.
    process_events();

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
      devproxy::freeHost(block->ptr_);
      delete block;
    }
  }

  bool is_pinned_ptr(const void* ptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return ptr_to_block_.find(const_cast<void*>(ptr)) != ptr_to_block_.end();
  }

 private:
  void process_events() {
    while (true) {
      // Avoid calling destroyEvent while holding a mutex, so move
      // intermediate events out of the lock into this object.
      c10::optional<std::pair<DIPUEvent, Block*>> processed;

      {
        std::lock_guard<std::mutex> g(events_mutex_);
        if (!events_.empty()) {
          processed = std::move(events_.back());
          events_.pop_back();
        }
      }

      if (!processed) {
        return;
      }

      // otherwise, query the event
      {
        // now, see if we can handle this element
        auto& event = processed->first;
        if (!event.query()) {
          std::lock_guard<std::mutex> g(events_mutex_);
          events_.push_back(std::move(*processed));
          return;
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

  alignas(kAlignPack) std::mutex blocks_mutex_;
  std::unordered_set<Block*> blocks_;
  std::unordered_map<void*, Block*> ptr_to_block_;
  // Note: sharding this mutex seems to be profitable in heavily multi-threaded
  // scenarios.
  alignas(kAlignPack) std::mutex free_list_mutex_;
  // Note: an alternative datastructure can yield significant wins here in
  // microbenchmarks.
  std::set<Block*, BlockComparator> free_list_;

  alignas(kAlignPack) std::mutex events_mutex_;
  std::deque<std::pair<DIPUEvent, Block*>> events_;
};

}  // namespace

static DIPUCachingHostAllocator& getDIPUCachingHostAllocator() {
  // leak and don't worry about shutdown
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static auto* r = new DIPUCachingHostAllocator();
  return *r;
}

static void DIPUCachingHostAllocatorDeleter(void* ctx) {
  getDIPUCachingHostAllocator().free(ctx);
}

bool CachingHostAllocator_recordEvent(void* ptr, void* ctx,
                                      const DIPUStream& stream) {
  return getDIPUCachingHostAllocator().record_event(ptr, ctx, stream);
}

// Releases cached pinned memory allocations via freeHost
void CachingHostAllocator_emptyCache() {
  getDIPUCachingHostAllocator().empty_cache();
}

bool CachingHostAllocator_isPinnedPtr(const void* ptr) {
  return getDIPUCachingHostAllocator().is_pinned_ptr(ptr);
}

struct DIPUCachingHostAllocatorWrapper final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    auto ptr_and_ctx = getDIPUCachingHostAllocator().allocate(size);
    return {ptr_and_ctx.first, ptr_and_ctx.second,
            &DIPUCachingHostAllocatorDeleter, at::DeviceType::CPU};
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static DIPUCachingHostAllocatorWrapper dipu_host_allocator;

at::Allocator* getCachingHostAllocator() { return &dipu_host_allocator; }

}  // namespace dipu::allocator
