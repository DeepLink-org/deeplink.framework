// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include <queue>
#include <vector>
#include <stack>
#include <atomic>
#include <thread>
#include <map>
#include <functional>

namespace dipu {

/// Simple spin-lock to help build thread-safe functions.
class SpinMutex {
private:
    std::atomic<bool> excl_ { false };

public:
    constexpr SpinMutex() noexcept = default;

    SpinMutex(const SpinMutex&) = delete;

    void delay() const noexcept {
        std::this_thread::yield();
    }

    void lock() {
        for (bool exp = false;
             !excl_.compare_exchange_weak(exp, true, std::memory_order_acq_rel);
             exp = false) delay();
    }

    bool try_lock() {
        bool exp = false;
        return
            excl_.compare_exchange_weak(exp, true, std::memory_order_acq_rel);
    }

    void unlock() {
        excl_.store(false, std::memory_order_release);
    }
};

class BFCachingAllocatorImpl{
public:
    using allocate_fn_t = std::function<void*(size_t)>;
    using deallocate_fn_t = std::function<void(void*)>;
private:
    allocate_fn_t allocate_fn;
    deallocate_fn_t deallocate_fn;
    // Number of first level bins (exponentially)
    static constexpr int kNumBigBins = 32;
    // Number of second level bins (linearly)
    static constexpr int kNumSubBins = 4;
    static constexpr int kLogNumSubBins = 2;
    // Allocation parameters
    static constexpr size_t kMinAllocationSize = 512;
    static constexpr size_t kMaxInternalFragmentation = 8u << 20u;       // 8MB
    static constexpr size_t kMinExtendSize = 8u << 20u;  // 8MB
    static constexpr size_t kMaxExtendSize = 1u << 30u;  // 1GB


    size_t cachedBytes = 0;
    size_t allocatedBytes = 0;

    void* allocateOnDevice(size_t nbytes) {
        void* ptr = nullptr;
        for (size_t i = 0; i < 2; i++) {
            try {
                ptr = allocate_fn(nbytes);
                cachedBytes += nbytes;
                break;
            }
            catch(...) {
                if (i == 0) {
                    emptyCacheWithoutLock();
                } else {
                    throw std::runtime_error("no device memory available");
                }
            }
        }
        DIPU_DEBUG_ALLOCATOR(4, "BFCachingAllocatorImpl: allocateOnDevice " << nbytes << " nbytes, ptr:" << ptr);
        return ptr;
    }

    void releaseOnDevice (void* ptr, size_t nbytes) {
        DIPU_DEBUG_ALLOCATOR(4, "BFCachingAllocatorImpl: releaseOnDevice " << nbytes << " nbytes, ptr:" << ptr);
        deallocate_fn(ptr);
        cachedBytes -= nbytes;
    }

    // Chunks and bins obtained by a single stream
    struct StreamSet {
        size_t id;
        // Compress whether bins have chunks
        // into 128 bits (`kNumBigBins` * `kNumSubBins`)
        __uint128_t bits = 0;
        // Virtual chunks which are the heads of the bins
        int binHeads_[kNumBigBins * kNumSubBins] {0};
        // The extending size next time
        size_t currExtendSize_ = kMinExtendSize;

        explicit StreamSet(size_t id): id(id) {}

        // Find an available bin greater than or equal to `least`
        int find(int least) const {
            // For the case that `least` >= 128,
            // the code below can also handle, we don't have to judge
            // Use `mask` to set the bits (< `least`) to 0
            __uint128_t mask = 1;
            (mask <<= least) -= 1;
            __uint128_t map = (bits | mask) ^ mask;
            // Find the index of the first "1"
            // `__builtin_ctzll` only support `uint64_t`,
            // so we have to divide
            uint64_t low_bits = map, high_bits = map >> 64u;
            if (low_bits) {
                return __builtin_ctzll(low_bits);
            }
            if (high_bits) {
                return 64 + __builtin_ctzll(high_bits);
            }
            return -1;
        }

        // Set `idx` bit into 1
        void set(unsigned idx) {
            __uint128_t mask = 1;
            mask <<= idx;
            bits |= mask;
        }

        // Set `idx` bit into 0
        void remove(unsigned idx) {
            __uint128_t mask = 1;
            mask <<= idx;
            bits &= ~mask;
        }
    };

    struct Chunk {
        bool allocated = false;
        int binId = -1;
        int prevChunkInMem = 0, nextChunkInMem = 0;
        int prevChunkInList = 0, nextChunkInList = 0;

        void* ptr;
        size_t size;
        // The stream id when created
        size_t stream;

        Chunk(void* ptr, size_t size, size_t stream):
                ptr(ptr), size(size), stream(stream) {}

        bool isMonoBlock() const {
            return !prevChunkInMem && !nextChunkInMem;
        }
    };

    std::vector<Chunk> chunks_;
    // Use id recycling for better performance
    std::stack<int> recycleIds_;

    typedef std::unique_ptr<StreamSet> StreamSetHandle;
    std::vector<StreamSetHandle> streamSets_;

    using mutex_t = SpinMutex;
    mutable mutex_t mut_;

    static size_t roundBytes(size_t nbytes) {
        return ((nbytes - 1) | (kMinAllocationSize - 1)) + 1;
    }

    int newChunk(void* ptr, size_t size, size_t stream) {
        int id;
        if (!recycleIds_.empty()) {
            id = recycleIds_.top();
            recycleIds_.pop();
            chunks_[id] = Chunk(ptr, size, stream);
        } else {
            id = chunks_.size();
            chunks_.emplace_back(Chunk(ptr, size, stream));
        }
        if (!ptr) {
            chunks_[id].allocated = true;
        }
        return id;
    }

    static int binIdForSize(size_t nbytes) {
        // Big bin range:
        //      [2^`bigBinIdx`, 2^(`bigBinIdx`+1)), length: 2^`bigBinIdx`
        // Split big bin into `kNumSubBins` sub bins
        size_t nBlocks = nbytes / kMinAllocationSize;
        int bigBinIdx = 63 - __builtin_clzll(nBlocks);
        // If `nbytes` is so large, we just put it into the last
        if (bigBinIdx > kNumBigBins - 1)
            return kNumBigBins * kNumSubBins - 1;
        // Get the index of sub bin
        int subBinIdx = nBlocks ^ (1ull << bigBinIdx);
        subBinIdx >>= std::max(bigBinIdx - kLogNumSubBins, 0);
        return bigBinIdx * kNumSubBins + subBinIdx;
    }


    void linkChunkInList(int a, int b, int c) {
        chunks_[a].nextChunkInList = b;
        chunks_[b].prevChunkInList = a;
        chunks_[b].nextChunkInList = c;
        chunks_[c].prevChunkInList = b;
    }

    void linkChunkInMem(int a, int b, int c) {
        chunks_[a].nextChunkInMem = b;
        chunks_[b].prevChunkInMem = a;
        chunks_[b].nextChunkInMem = c;
        chunks_[c].prevChunkInMem = b;
    }

    void removeChunkInList(int a, int c) {
        // Remove b
        chunks_[a].nextChunkInList = c;
        chunks_[c].prevChunkInList = a;
    }

    void removeChunkInMem(int a, int c) {
        // Remove b
        chunks_[a].nextChunkInMem = c;
        chunks_[c].prevChunkInMem = a;
    }

    void insertChunkIntoBin(int id) {
        int binId = (chunks_[id].binId = binIdForSize(chunks_[id].size));
        auto &set = streamSets_[chunks_[id].stream];
        set->set(binId);
        linkChunkInList(set->binHeads_[binId], id,
                chunks_[set->binHeads_[binId]].nextChunkInList);
    }

    void removeChunkFromBin(int id) {
        int binId = chunks_[id].binId;
        auto &set = streamSets_[chunks_[id].stream];
        removeChunkInList(chunks_[id].prevChunkInList,
                chunks_[id].nextChunkInList);
        if (!chunks_[set->binHeads_[binId]].nextChunkInList) {
            set->remove(binId);
        }
    }

    int findChunk(size_t nbytes, StreamSetHandle &set) {
        // Check whether the first chunk in `least` bin satisfies
        int least = binIdForSize(nbytes);
        int id = chunks_[set->binHeads_[least]].nextChunkInList;
        if (id) {
            id = chunks_[id].size >= nbytes ? id : 0;
        }

        // If not, check the next available bin
        if (!id) {
            int binId = set->find(least + 1);
            id = (binId == -1) ?
                 0 : chunks_[set->binHeads_[binId]].nextChunkInList;
        }

        if (id) {
            removeChunkFromBin(id);
        }
        return id;
    }

    void shrink(StreamSetHandle &set) {
        for (int binHead : set->binHeads_) {
            int k = chunks_[binHead].nextChunkInList;
            while (k) {
                if (chunks_[k].isMonoBlock()) {
                    releaseOnDevice(chunks_[k].ptr, chunks_[k].size);
                    removeChunkFromBin(k);
                    recycleIds_.push(k);
                }
                k = chunks_[k].nextChunkInList;
            }
        }
    }

    int split(int id, size_t nbytes) {
        void* ptr = static_cast<char*>(chunks_[id].ptr) + nbytes;
        size_t const size = chunks_[id].size - nbytes;

        chunks_[id].size = nbytes;

        int newId = newChunk(ptr, size, chunks_[id].stream);
        linkChunkInMem(id, newId, chunks_[id].nextChunkInMem);
        insertChunkIntoBin(newId);

        return id;
    }

    int merge(int c1, int c2) {
        chunks_[c1].size += chunks_[c2].size;
        removeChunkInMem(c1, chunks_[c2].nextChunkInMem);
        return c1;
    }

    int coalesce(int id) {
        int next = chunks_[id].nextChunkInMem;
        if (next && !chunks_[next].allocated) {
            removeChunkFromBin(next);
            id = merge(id, next);
            recycleIds_.push(next);
        }

        int prev = chunks_[id].prevChunkInMem;
        if (prev && !chunks_[prev].allocated) {
            removeChunkFromBin(prev);
            int oldId = id;
            id = merge(prev, id);
            recycleIds_.push(oldId);
        }

        return id;
    }

    int extend(size_t nbytes, StreamSetHandle &set) {
        emptyCacheWithoutLock();
        auto& extSize = set->currExtendSize_;
        bool increased = false;
        while (extSize < nbytes && extSize < kMaxExtendSize) {
            extSize *= 2;
            increased = true;
        }

        size_t currBytes = std::max(nbytes, extSize);
        void* ptr = allocateOnDevice(currBytes);
        if (ptr) {
            if (!increased && extSize < kMaxExtendSize) {
                extSize *= 2;
            }
        } else {
            if (currBytes > nbytes) {
                currBytes = nbytes;
                ptr = allocateOnDevice(currBytes);
            }
        }
        if (!ptr) {
            return 0;
        }

        int id = newChunk(ptr, currBytes, set->id);
        return id;
    }

    StreamSetHandle& checkStream(size_t stream) {
        if (stream >= streamSets_.size()) {
            streamSets_.resize(stream + 1);
        }
        if (streamSets_[stream] == nullptr) {
            streamSets_[stream] = std::make_unique<StreamSet>(stream);
            for (int &binHead : streamSets_[stream]->binHeads_) {
                binHead = newChunk(nullptr, 0, 0);
            }
        }
        return streamSets_[stream];
    }

    void emptyCacheWithoutLock() {
        for (auto &set : streamSets_) {
            if (set != nullptr) {
                shrink(set);
            }
        }
    }

public:
    BFCachingAllocatorImpl() {
        // Avoid zero index later
        newChunk(nullptr, 0, 0);
    }

    ~BFCachingAllocatorImpl() {
        emptyCache();
    }

    void emptyCache() {
        std::lock_guard<mutex_t> lk(mut_);
        emptyCacheWithoutLock();
    }

    std::pair<void*, int> allocateRaw(size_t nbytes) {
        if (!nbytes) {
            return std::make_pair(nullptr, 0);
        }

        nbytes = roundBytes(nbytes);

        allocatedBytes += nbytes;

        std::lock_guard<mutex_t> lk(mut_);
        auto &set = checkStream(0);
        int id = findChunk(nbytes, set);
        if (!id) {
            id = extend(nbytes, set);
        }

        if (id) {
            if (chunks_[id].size >= nbytes * 2 ||
                chunks_[id].size >= nbytes + kMaxInternalFragmentation) {
                id = split(id, nbytes);
            }
            chunks_[id].allocated = true;
            return std::make_pair(chunks_[id].ptr, id);
        }
        return std::make_pair(nullptr, 0);;
    }

    void releaseRaw(void* ptr, int id) {
        if (!ptr) {
            return;
        }

        std::lock_guard<mutex_t> lk(mut_);
        chunks_[id].allocated = false;
        allocatedBytes -= chunks_[id].size;
        id = coalesce(id);
        insertChunkIntoBin(id);
    }

    void set_mem_allocate_fn(allocate_fn_t allocate_fn, deallocate_fn_t deallocate_fn) {
         DIPU_DEBUG_ALLOCATOR(4, "BFCachingAllocator: set_mem_allocate_fn ");
        this->allocate_fn = allocate_fn;
        this->deallocate_fn = deallocate_fn;
    }
};

static void deleteBFContext(void* ptr);

class BFCachingAllocator: public CacheAllocator {
    mutable std::unique_ptr<BFCachingAllocatorImpl> impl;

private:
  void restore() const{
    while (async_mem_pool()->ready()) {
        const auto block = async_mem_pool()->get();
        void* ptr = std::get<0>(block);
        int id = std::get<1>(block);
        DIPU_DEBUG_ALLOCATOR(8, "BFCachingAllocator: " << __FUNCTION__ << " ,ptr:" << ptr << " ,id:" << id << " ,allocator:" << this << ", device:" << device());
        impl->releaseRaw(ptr, id);
    }
  }

  void check_impl() const{
    if (impl) {
        return;
    }
    impl.reset(new BFCachingAllocatorImpl());

    std::function<void*(size_t)> alloc_fn = std::bind(&BFCachingAllocator::allocate_raw, (BFCachingAllocator*)this, std::placeholders::_1);
    std::function<void(void*)> dealloc_fn = std::bind(&BFCachingAllocator::free_raw, (BFCachingAllocator*)this, std::placeholders::_1);
    impl->set_mem_allocate_fn(alloc_fn, dealloc_fn);
  }

  void* makeContext(void* ptr, size_t size, int id) const{
        auto ctx = new Context(ptr, size, id, this);
        return ctx;
  }

public:
  struct Context: public DataPtrContextBase {
    int id_;
    Context(void* ptr, size_t size, int id, const BFCachingAllocator* allocator):DataPtrContextBase(allocator, ptr, size), id_(id) {

    }

    ~Context() {
      auto allocator_ = static_cast<const BFCachingAllocator*>(allocator());
      DIPU_DEBUG_ALLOCATOR(8, "BFCachingAllocator: add to async_mem_pool:" << ptr() << ", " << size() << " nbytes, id:"<< id_ <<", allocator:" << allocator_ << ", device:" << allocator_->device());
      if (allocator_->impl) {
        if (ptr()) {
            std::deque<DIPUEvent> events;
            for (auto iter = streams().begin(); iter != streams().end(); iter++) {
                events.emplace_back();
                DIPU_DEBUG_ALLOCATOR(8, "BFCachingAllocator: record to stream:" << iter->rawstream() );
                events.back().record(*iter);
            }
            allocator_->async_mem_pool()->add(std::make_tuple(ptr(), id_), events);
        }
        allocator_->restore();
      } else {
        DIPU_DEBUG_ALLOCATOR(8, "BFCachingAllocator:~Context: destory tensor when allocator has been destoryed");
      }
    }
  };


  c10::DataPtr allocate(size_t size) const override {
    check_impl();
    std::pair<void*, int> block = impl->allocateRaw(size);
    void* ptr = std::get<0>(block);
    int id = std::get<1>(block);

    c10::DataPtr data_ptr(ptr, makeContext(ptr, size, id), deleteBFContext, device());
    DIPU_DEBUG_ALLOCATOR(4, "BFCachingAllocator: malloc " << size << " nbytes, ptr:" << ptr << ",device:" << device());
    return data_ptr;
  }

  void empty_cache() const override {
    DIPU_DEBUG_ALLOCATOR(8, "BFCachingAllocator: empty_cache, allocator:" << this << ", device:" << device());
    impl->emptyCache();
  }

  void release_all_memory() const override {
    if (!impl) {
        return;
    }
    DIPU_DEBUG_ALLOCATOR(8, "BFCachingAllocator: release_all_memory, allocator:" << this << ", device:" << device());
    while (async_mem_pool()->size() > 0) {
        if (!async_mem_pool()->ready()) {
            std::this_thread::yield();
            continue;
        }
        const auto block = async_mem_pool()->get();
        void* ptr = std::get<0>(block);
        int id = std::get<1>(block);
        impl->releaseRaw(ptr, id);
    }
    impl.reset(nullptr);
  }

  BFCachingAllocator() {

  }

  ~BFCachingAllocator() {
    release_all_memory();
  }

};

static void deleteBFContext(void* ptr) {
  auto ctx = static_cast<BFCachingAllocator::Context*>(ptr);
  delete ctx;
}

DIPU_REGISTER_ALLOCATOR(BF, dipu::DIPU_DEVICE_TYPE, BFCachingAllocator, 0);
DIPU_REGISTER_ALLOCATOR(BF, at::DeviceType::CPU, BFCachingAllocator, 0);



}  // namespace dipu
