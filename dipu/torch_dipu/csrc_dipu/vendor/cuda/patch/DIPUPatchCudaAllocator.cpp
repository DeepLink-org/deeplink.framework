#include <mutex>
#include <unordered_map>

#include "c10/cuda/CUDACachingAllocator.h"

#include <csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h>

namespace c10 {

namespace cuda {

namespace CUDACachingAllocator {

#define DIPU_PATCH_CUDA_ALLOCATOR(x)           \
  std::cout << __FUNCTION__ << ":" << __LINE__ \
            << " this function should not be called!" x << std::endl;

class DIPUCUDAAllocatorProxy : public CUDAAllocator {
  std::unordered_map<void*, c10::DataPtr> tempMemBlock;
  using mutex_t = std::mutex;
  mutable mutex_t mut_;

 public:
  virtual void* raw_alloc_with_stream(size_t nbytes,
                                      cudaStream_t stream) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual void setMemoryFraction(double fraction, int device) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual void* getBaseAllocation(void* ptr, size_t* size) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual void recordStream(const DataPtr&, CUDAStream stream) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual DeviceStats getDeviceStats(int device) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual void resetAccumulatedStats(int device) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual void resetPeakStats(int device) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual SnapshotInfo snapshot() override { DIPU_PATCH_CUDA_ALLOCATOR(); }
  void notifyCaptureBegin(int device, CaptureId_t graph_id,
                          MempoolId_t mempool_id) {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void notifyCaptureEnded(int device, CaptureId_t graph_id) {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void notifyCaptureDestroy(int device, MempoolId_t mempool_id) {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries,
                     bool alloc_trace_record_context) {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual void attachOutOfMemoryObserver(
      OutOfMemoryObserver observer) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }
  virtual std::string name() override { DIPU_PATCH_CUDA_ALLOCATOR(); }
  virtual void cacheInfo(int dev_id, size_t* largestBlock) override {
    DIPU_PATCH_CUDA_ALLOCATOR();
  }

  virtual void* raw_alloc(size_t nbytes) override {
    auto data_ptr = this->allocate(nbytes);
    void* ptr = data_ptr.get();
    std::lock_guard<mutex_t> lk(mut_);
    tempMemBlock.emplace(ptr, std::move(data_ptr));
    return ptr;
  }

  virtual void raw_delete(void* ptr) override {
    std::lock_guard<mutex_t> lk(mut_);
    tempMemBlock.erase(ptr);
  }

  virtual void init(int device_count) override {}

  virtual bool initialized() override { return true; }

  virtual void emptyCache() override { dipu::emptyCachedMem(); }

  bool needsPoolSpecificPeerAccess() {
    // DIPU_PATCH_CUDA_ALLOCATOR();
    return false;
  }

  virtual DataPtr allocate(size_t n) const override {
    // DIPU_PATCH_CUDA_ALLOCATOR();
    auto data_ptr = c10::GetAllocator(dipu::DIPU_DEVICE_TYPE)->allocate(n);
    data_ptr.unsafe_set_device(
        c10::Device(c10::DeviceType::CUDA, data_ptr.device().index()));
    return data_ptr;
  }

  void beginAllocateStreamToPool(int device, cudaStream_t stream,
                                 MempoolId_t mempool_id) {}
  void endAllocateStreamToPool(int device, cudaStream_t stream) {}

  void recordHistory(bool enabled, CreateContextFn context_recorder,
                     size_t alloc_trace_max_entries, RecordContext when) {}
  void releasePool(int device, MempoolId_t mempool_id) {}

  void enablePeerAccess(int dev, int dev_to_access) {}

  cudaError_t memcpyAsync(void* dst, int dstDevice, const void* src,
                          int srcDevice, size_t count, cudaStream_t stream,
                          bool p2p_enabled) {}
  std::shared_ptr<AllocatorState> getCheckpointState(int device,
                                                     MempoolId_t id) {}
  CheckpointDelta setCheckpointPoolState(int device,
                                         std::shared_ptr<AllocatorState> pps) {
    return CheckpointDelta();
  }
};

}  // namespace CUDACachingAllocator

}  // namespace cuda

}  // namespace c10

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
static int n = patchCachingAllocator();

}  // namespace dipu
