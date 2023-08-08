#include "DIPUCachingAllocator.h"
#include "c10/cuda/CUDACachingAllocator.h"

namespace c10 {

namespace cuda {

namespace CUDACachingAllocator {

#define DIPU_PATCH_CUDA_ALLOCATOR(x) \
    std::cout << __FUNCTION__ << ":" << __LINE__ << "this function should not be called!" x << std::endl;

class DIPUCUDAAllocatorProxy : public CUDAAllocator {
 public:
  virtual void* raw_alloc(size_t nbytes) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void raw_delete(void* ptr) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void init(int device_count) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual bool initialized() override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void setMemoryFraction(double fraction, int device) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void emptyCache() override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void cacheInfo(int dev_id, size_t* largestBlock) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void* getBaseAllocation(void* ptr, size_t* size) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void recordStream(const DataPtr&, CUDAStream stream) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual DeviceStats getDeviceStats(int device) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void resetAccumulatedStats(int device) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void resetPeakStats(int device) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual SnapshotInfo snapshot() override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void notifyCaptureBegin(int device, CaptureId_t graph_id, MempoolId_t mempool_id) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void notifyCaptureAboutToEnd(int device, CaptureId_t graph_id) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void notifyCaptureEnded(int device, CaptureId_t graph_id) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void notifyCaptureDestroy(int device, MempoolId_t mempool_id) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void recordHistory(bool enabled, CreateContextFn context_recorder, size_t alloc_trace_max_entries, bool alloc_trace_record_context) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual void attachOutOfMemoryObserver(OutOfMemoryObserver observer) override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual std::string name() override { DIPU_PATCH_CUDA_ALLOCATOR();}
  virtual bool needsPoolSpecificPeerAccess() override {
    // DIPU_PATCH_CUDA_ALLOCATOR();
    return false;
  }
  
  virtual DataPtr allocate(size_t n) const override {
    //DIPU_PATCH_CUDA_ALLOCATOR();
    auto data_ptr = c10::GetAllocator(dipu::DIPU_DEVICE_TYPE)->allocate(n);
    data_ptr.unsafe_set_device(c10::Device(c10::DeviceType::CUDA, data_ptr.device().index()));
    return data_ptr;
  }
};

static DIPUCUDAAllocatorProxy cuda_allocator_proxy;

} // namespace CUDACachingAllocator

} // namespace cuda

} // namespace c10

int patchCachingAllocator() {
  std::cout << __FUNCTION__ << std::endl;
  c10::cuda::CUDACachingAllocator::allocator.store(dynamic_cast<c10::cuda::CUDACachingAllocator::CUDAAllocator*>(&c10::cuda::CUDACachingAllocator::cuda_allocator_proxy));
  return 0;
}

static int m = patchCachingAllocator();
