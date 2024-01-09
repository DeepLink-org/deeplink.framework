#include <mutex>
#include <unordered_map>

#include "c10/cuda/CUDACachingAllocator.h"

#include <csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h>

namespace c10::cuda::CUDACachingAllocator {

#define DIPU_PATCH_CUDA_ALLOCATOR(x)                                  \
  std::cout << __FUNCTION__ << ":" << __LINE__                        \
            << " this function should not be called!" x << std::endl; \
  throw std::runtime_error("this function should not be called!");

class DIPUCUDAAllocatorProxy : public CUDAAllocator {
  std::unordered_map<void*, c10::DataPtr> tempMemBlock;
  using mutex_t = std::mutex;
  mutable mutex_t mut_;

 public:
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void setMemoryFraction(double fraction, int device) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void* getBaseAllocation(void* ptr, size_t* size) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void recordStream(const DataPtr& /*unused*/, CUDAStream stream) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  DeviceStats getDeviceStats(int device) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void resetAccumulatedStats(int device) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void resetPeakStats(int device) override { DIPU_PATCH_CUDA_ALLOCATOR(); }
  SnapshotInfo snapshot() override { DIPU_PATCH_CUDA_ALLOCATOR(); }

  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  std::string name() override { DIPU_PATCH_CUDA_ALLOCATOR(); }
  void cacheInfo(int dev_id, size_t* largestBlock) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }

  void* raw_alloc(size_t nbytes) override {
    auto data_ptr = this->allocate(nbytes);
    void* ptr = data_ptr.get();
    std::lock_guard<mutex_t> lk(mut_);
    tempMemBlock.emplace(ptr, std::move(data_ptr));
    return ptr;
  }

  void raw_delete(void* ptr) override {
    std::lock_guard<mutex_t> lk(mut_);
    tempMemBlock.erase(ptr);
  }

  void init(int device_count) override {}

  bool initialized() override { return true; }

  void emptyCache() override { dipu::emptyCachedMem(); }

  DataPtr allocate(size_t n) const override {
    // DIPU_PATCH_CUDA_ALLOCATOR();
    auto data_ptr = c10::GetAllocator(dipu::DIPU_DEVICE_TYPE)->allocate(n);
    data_ptr.unsafe_set_device(
        c10::Device(c10::DeviceType::CUDA, data_ptr.device().index()));
    return data_ptr;
  }
#ifdef DIPU_TORCH200
  void notifyCaptureBegin(int device, CaptureId_t graph_id,
                          MempoolId_t mempool_id) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void notifyCaptureEnded(int device, CaptureId_t graph_id) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void notifyCaptureDestroy(int device, MempoolId_t mempool_id) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }

  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries,
                     bool alloc_trace_record_context) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  bool needsPoolSpecificPeerAccess() override {
    // DIPU_PATCH_CUDA_ALLOCATOR();
    return false;
  }

#else  // # DIPU_TORCH211 or higher
  void beginAllocateStreamToPool(int device, cudaStream_t stream,
                                 MempoolId_t mempool_id) override {}
  void endAllocateStreamToPool(int device, cudaStream_t stream) override {}

  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries,
                     RecordContext when) override {}
  void releasePool(int device, MempoolId_t mempool_id) override {}

  void enablePeerAccess(int dev, int dev_to_access) override {}

  cudaError_t memcpyAsync(void* dst, int dstDevice, const void* src,
                          int srcDevice, size_t count, cudaStream_t stream,
                          bool p2p_enabled) override {
    return cudaSuccess;
  }
  std::shared_ptr<AllocatorState> getCheckpointState(int device,
                                                     MempoolId_t id) override {}
  CheckpointDelta setCheckpointPoolState(
      int device, std::shared_ptr<AllocatorState> pps) override {
    return {};
  }
#endif

};  // namespace CUDACachingAllocator

}  // namespace c10::cuda::CUDACachingAllocator

namespace dipu {

int patchCachingAllocator() {
  const char* env = std::getenv("DIPU_PATCH_CUDA_CACHED_ALLOCATOR");
  if (env != nullptr) {
    if (std::atoi(env) <= 0) {
      return 0;
    }
  } else {
    return 0;
  }
  /*
    Our implementation idea is different from the native pytorch implementation,
    so the interface cannot be fully aligned. We only implement the most basic
    and necessary functions.
  */
  static c10::cuda::CUDACachingAllocator::DIPUCUDAAllocatorProxy
      cuda_allocator_proxy;
  c10::cuda::CUDACachingAllocator::allocator.store(
      dynamic_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator*>(
          &cuda_allocator_proxy));
  return 0;
}
/*This order is really unrequired and unimportant,
and this compilation unit may not be compiled, so it is still initialized with
global variables
*/
static const int n = patchCachingAllocator();

}  // namespace dipu
