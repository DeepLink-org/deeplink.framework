// Copyright (c) 2023, DeepLink.

#include "DIPUCachingAllocator.h"
#include <queue>
#include <vector>
#include <stack>
#include <atomic>
#include <thread>
#include <map>

namespace dipu {

/// Simple spin-lock to help build thread-safe functions.
class SpinMutex {
private:
    std::atomic<bool> excl_ { false };

public:
    constexpr SpinMutex() noexcept = default;

    SpinMutex(const SpinMutex&) = delete;

    void delay() const noexcept {
        // TODO(lizhouyang): provide more delay mod: spin, yield, sleep.
        // Just do nothing but burning cpu right here.
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

class BFCachingAllocator: public CacheAllocator {
private:
    // Number of first level bins (exponentially)
    static constexpr int kNumBigBins = 32;
    // Number of second level bins (linearly)
    static constexpr int kNumSubBins = 4;
    static constexpr int kLogNumSubBins = 2;
    // Allocation parameters
    static constexpr size_t kMinAllocationSize = 512;
    static constexpr size_t kMaxInternalFragmentation = 8u << 20u;       // 8MB
    static constexpr size_t kMinExtendSize = kMinAllocationSize << 14u;  // 8MB
    static constexpr size_t kMaxExtendSize = kMinAllocationSize << 21u;  // 1GB

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
        // The chunks going to be released
        std::queue<int> waiting;

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
        // In the future which can be released
        int future;

        Chunk(void* ptr, size_t size, size_t stream):
                ptr(ptr), size(size), stream(stream) {}

        bool isMonoBlock() const {
            return !prevChunkInMem && !nextChunkInMem;
        }
    };

    mutable std::vector<Chunk> chunks_;
    // Use id recycling for better performance
    std::stack<int> recycleIds_;

    typedef std::unique_ptr<StreamSet> StreamSetHandle;
    mutable std::vector<StreamSetHandle> streamSets_;

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
        TORCH_CHECK(nBlocks > 0);
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

    void checkWaiting(StreamSetHandle &set, bool wait) {
        auto& queue = set->waiting;
        while (!queue.empty()) {
            int id = queue.front();
            // TODO
            #if 0
            if (wait) {
                chunks_[id].future.wait();
            }
            if (!chunks_[id].future.isReady()) {
                break;
            }
            #else
            break;
            #endif

            queue.pop();
            chunks_[id].allocated = false;
            id = coalesce(id);
            insertChunkIntoBin(id);
        }
    }

    int findChunk(size_t nbytes, StreamSetHandle &set) {
        checkWaiting(set, false);

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
        checkWaiting(set, true);
        for (int binHead : set->binHeads_) {
            int k = chunks_[binHead].nextChunkInList;
            while (k) {
                if (chunks_[k].isMonoBlock()) {
                    // TODO
                    //chunks_[k].future.wait();
                    //releaseOnDevice(chunks_[k].ptr, chunks_[k].size);
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
        chunks_[newId].future = chunks_[id].future;
        linkChunkInMem(id, newId, chunks_[id].nextChunkInMem);
        insertChunkIntoBin(newId);

        return id;
    }

    int merge(int c1, int c2) {
        chunks_[c1].size += chunks_[c2].size;
        #if 0
        if (chunks_[c1].future.streamId() == chunks_[c2].future.streamId()) {
            chunks_[c1].future =
                    parrots::merge(chunks_[c1].future, chunks_[c2].future);
        } else {
            if (chunks_[c1].future.isReady()) {
                chunks_[c1].future = chunks_[c2].future;
            } else {
                PARROTS_CHECKARGS(chunks_[c2].future.isReady()) <<
                        "Unable to merge Future from different stream.";
            }
        }
        #endif

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

    #if 0
    int extend(size_t nbytes, StreamSetHandle &set) {
        if (nbytes > available()) {
            shrink(set);
        }

        if (nbytes > available()) {
            return 0;
        }

        auto& extSize = set->currExtendSize_;
        bool increased = false;
        while (extSize < nbytes && extSize < kMaxExtendSize) {
            extSize *= 2;
            increased = true;
        }

        size_t currBytes = std::min(std::max(nbytes, extSize), available());
        void* ptr = allocate_raw(currBytes);
        if (ptr) {
            if (!increased && extSize < kMaxExtendSize) {
                extSize *= 2;
            }
        } else {
            if (currBytes > nbytes) {
                currBytes = nbytes;
                ptr = allocate_raw(currBytes);
            }
        }
        if (!ptr) {
            return 0;
        }

        int id = newChunk(ptr, currBytes, set->id);
        return id;
    }
    #endif
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


public:
  BFCachingAllocator() {
    // Avoid zero index later
    newChunk(nullptr, 0, 0);
  }

  ~BFCachingAllocator() {
    empty_cache();
    for (auto &set : streamSets_) {
        if (set != nullptr) {
             TORCH_CHECK(set->find(0) == -1, "Stream ", set->id ," has unreleased memory");
        }
    }
  }


    c10::DataPtr allocate(size_t nbytes) const {

    }

    void empty_cache() const override {
    std::lock_guard<mutex_t> lk(mut_);
        for (auto &set : streamSets_) {
            if (set != nullptr) {
                //shrink(set);
            }
        }
    }

    void release_all_memory() const override {

    }

#if 0

   c10::DataPtr allocate_ptr(size_t nbytes) {
    if (!nbytes) {
            return c10::DataPtr();
    }

        nbytes = roundBytes(nbytes);

        std::lock_guard<mutex_t> lk(mut_);
        //auto &set = checkStream(future.streamId());
        // TODO:
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
            // TODO
            #if 0
            if (chunks_[id].future.streamId() != future.streamId()) {
                chunks_[id].future.wait();
            }
            #endif

            chunks_[id].allocated = true;
            return makeDataPtr(chunks_[id].ptr);
        }
        return c10::DataPtr();
  }
    c10::DataPtr makeDataPtr(void* ptr) const {

    }

#endif
};

DIPU_REGISTER_ALLOCATOR(BF, dipu::DIPU_DEVICE_TYPE, BFCachingAllocator, 0);
DIPU_REGISTER_ALLOCATOR(BF, at::DeviceType::CPU, BFCachingAllocator, 0);

#if 0

class MbfMemoryManager final : public CachedMemoryManager {
private:
    // Number of first level bins (exponentially)
    static constexpr int kNumBigBins = 32;
    // Number of second level bins (linearly)
    static constexpr int kNumSubBins = 4;
    static constexpr int kLogNumSubBins = 2;
    // Allocation parameters
    static constexpr size_t kMinAllocationSize = PARROTS_ALIGNMENT;
    static constexpr size_t kMaxInternalFragmentation = 8u << 20u;       // 8MB
    static constexpr size_t kMinExtendSize = kMinAllocationSize << 14u;  // 8MB
    static constexpr size_t kMaxExtendSize = kMinAllocationSize << 21u;  // 1GB

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
        // The chunks going to be released
        std::queue<int> waiting;

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
        // In the future which can be released
        Future future;

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
        PARROTS_ASSERT(nBlocks > 0);
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

    void checkWaiting(StreamSetHandle &set, bool wait) {
        auto& queue = set->waiting;
        while (!queue.empty()) {
            int id = queue.front();
            if (wait) {
                chunks_[id].future.wait();
            }
            if (!chunks_[id].future.isReady()) {
                break;
            }
            queue.pop();
            chunks_[id].allocated = false;
            id = coalesce(id);
            insertChunkIntoBin(id);
        }
    }

    int findChunk(size_t nbytes, StreamSetHandle &set) {
        checkWaiting(set, false);

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
        checkWaiting(set, true);
        for (int binHead : set->binHeads_) {
            int k = chunks_[binHead].nextChunkInList;
            while (k) {
                if (chunks_[k].isMonoBlock()) {
                    chunks_[k].future.wait();
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
        chunks_[newId].future = chunks_[id].future;
        linkChunkInMem(id, newId, chunks_[id].nextChunkInMem);
        insertChunkIntoBin(newId);

        return id;
    }

    int merge(int c1, int c2) {
        chunks_[c1].size += chunks_[c2].size;
        if (chunks_[c1].future.streamId() == chunks_[c2].future.streamId()) {
            chunks_[c1].future =
                    parrots::merge(chunks_[c1].future, chunks_[c2].future);
        } else {
            if (chunks_[c1].future.isReady()) {
                chunks_[c1].future = chunks_[c2].future;
            } else {
                PARROTS_CHECKARGS(chunks_[c2].future.isReady()) <<
                        "Unable to merge Future from different stream.";
            }
        }

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
        if (nbytes > available()) {
            shrink(set);
        }

        if (nbytes > available()) {
            return 0;
        }

        auto& extSize = set->currExtendSize_;
        bool increased = false;
        while (extSize < nbytes && extSize < kMaxExtendSize) {
            extSize *= 2;
            increased = true;
        }

        size_t currBytes = std::min(std::max(nbytes, extSize), available());
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
            streamSets_[stream] = clue::make_unique<StreamSet>(stream);
            for (int &binHead : streamSets_[stream]->binHeads_) {
                binHead = newChunk(nullptr, 0, 0);
            }
        }
        return streamSets_[stream];
    }

public:
    MbfMemoryManager(DeviceProxy& devProxy, size_t limit)
    : CachedMemoryManager(devProxy, limit) {
        // Avoid zero index later
        newChunk(nullptr, 0, 0);
    }

    ~MbfMemoryManager() {
        emptyCache();
        for (auto &set : streamSets_) {
            if (set != nullptr) {
                PARROTS_CHECKARGS_NT(set->find(0) == -1)
                        << "Stream " << set->id << " has unreleased memory";
            }
        }
    }

    void emptyCache() override {
        std::lock_guard<mutex_t> lk(mut_);
        for (auto &set : streamSets_) {
            if (set != nullptr) {
                shrink(set);
            }
        }
    }

    void synchronize() override {
        std::lock_guard<mutex_t> lk(mut_);
        for (auto& chunk : chunks_) {
            chunk.future.wait();
            chunk.future = Future();
        }
    }

    std::pair<void*, int> allocateRaw(
            size_t nbytes,
            const Future& future) override {
        if (!nbytes) {
            return std::make_pair(nullptr, 0);
        }

        nbytes = roundBytes(nbytes);

        std::lock_guard<mutex_t> lk(mut_);
        auto &set = checkStream(future.streamId());
        int id = findChunk(nbytes, set);
        if (!id) {
            id = extend(nbytes, set);
        }

        if (id) {
            if (chunks_[id].size >= nbytes * 2 ||
                chunks_[id].size >= nbytes + kMaxInternalFragmentation) {
                id = split(id, nbytes);
            }
            if (chunks_[id].future.streamId() != future.streamId()) {
                chunks_[id].future.wait();
            }

            chunks_[id].allocated = true;
            return std::make_pair(chunks_[id].ptr, id);
        }
        return std::make_pair(nullptr, 0);;
    }

    void releaseRaw(
            void* ptr,
            size_t nbytes,
            int id,
            const Future &future) override {
        if (!ptr) {
            return;
        }

        std::lock_guard<mutex_t> lk(mut_);
        chunks_[id].future = future;
        if (chunks_[id].future.streamId() == chunks_[id].stream) {
            chunks_[id].allocated = false;
            id = coalesce(id);
            insertChunkIntoBin(id);
        } else {
            streamSets_[chunks_[id].stream]->waiting.push(id);
        }
    }
};


#endif

}  // namespace dipu
