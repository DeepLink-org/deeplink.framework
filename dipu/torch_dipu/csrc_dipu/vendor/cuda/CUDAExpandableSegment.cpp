// Copyright (c) 2024, DeepLink.
#include <iostream>
#include <vector>

#include "csrc_dipu/runtime/core/allocator/ExpandableSegment.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include "csrc_dipu/vendor/cuda/basecuda.hpp"

namespace dipu {

// ----------------------------------------------------------------------------
// Code from pytorch2.1.1 c10/cuda/CUDACachingAllocator.cpp
// ----------------------------------------------------------------------------

class CUDAExpandableSegment : public ExpandableSegment {
 public:
  CUDAExpandableSegment(int device, deviceStream_t stream, size_t size,
                        std::vector<int> peers)
      : device_(device),
        stream_(stream),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size),
        peers_(std::move(peers)) {
    devapis::DIPUDeviceProperties prop = devproxy::getDeviceProperties(device_);
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(prop.totalGlobalMem + prop.totalGlobalMem / 8);
    DIPU_DRIVER_CHECK(cuMemAddressReserve(&ptr_, segment_size_ * max_handles_,
                                          0ULL, 0, 0ULL));
  }
  // begin must be aligned to segment_size_.
  // returns the actual range mapped, which may be
  // greater than requested if size is not aligned to segment_size_.
  // return size of 0 indicates OOM
  SegmentRange map(SegmentRange range) override {
    auto begin = segmentLeft(range.ptr);
    auto end = segmentRight(range.ptr + range.size);
    TORCH_INTERNAL_ASSERT(ptr() + begin * segment_size_ == range.ptr);
    if (begin == end) {
      return rangeFromHandles(begin, end);
    }
    while (end > handles_.size()) {
      handles_.emplace_back(c10::nullopt);
    }
    for (auto i : c10::irange(begin, end)) {
      TORCH_INTERNAL_ASSERT(!handles_.at(i));
      CUmemGenericAllocationHandle handle = 0;
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
      auto status = cuMemCreate(&handle, segment_size_, &prop, 0);
      if (status == CUDA_ERROR_OUT_OF_MEMORY) {
        for (auto j : c10::irange(begin, i)) {
          auto h = handles_.at(j).value();
          handles_.at(j) = c10::nullopt;
          DIPU_DRIVER_CHECK(cuMemRelease(h));
        }
        trimHandles();
        return rangeFromHandles(begin, begin);
      }
      DIPU_DRIVER_CHECK(status);
      handles_.at(i) = handle;
    }

    for (auto i : c10::irange(begin, end)) {
      DIPU_DRIVER_CHECK(cuMemMap(ptr_ + i * segment_size_, segment_size_, 0,
                                 handles_.at(i).value(), 0ULL));
    }
    setAccess(device_, begin, end);
    for (auto p : peers_) {
      setAccess(p, begin, end);
    }
    return rangeFromHandles(begin, end);
  }

  // unmaps all the completely empty segment_size_ segments between
  // [begin, begin + size), returns the offset where the range begin,
  // and the actual size unmapped (multiple of segment_size_)
  SegmentRange unmap(SegmentRange range) override {
    auto begin = segmentRight(range.ptr);
    auto end = segmentLeft(range.ptr + range.size);
    if (begin >= end) {
      return SegmentRange{range.ptr, 0};
    }
    unmapHandles(begin, end);
    return rangeFromHandles(begin, end);
  }

  char* ptr() const { return (char*)ptr_; }
  size_t size() const override { return max_handles_ * segment_size_; }
  void addPeer(int device) override {
    peers_.push_back(device);
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { setAccess(device, begin, end); });
  }

 public:
  ~CUDAExpandableSegment() noexcept override {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    DIPU_DRIVER_CHECK(cuMemAddressFree(ptr_, segment_size_ * max_handles_));
  }

 private:
  void setAccess(int device, size_t begin, size_t end) {
    CUmemAccessDesc desc;
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DIPU_DRIVER_CHECK(cuMemSetAccess(ptr_ + begin * segment_size_,
                                     (end - begin) * segment_size_, &desc, 1));
  }

  void unmapHandles(size_t begin, size_t end) {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    devproxy::syncStream(stream_);
    for (auto i : c10::irange(begin, end)) {
      // aclrtDrvMemHandle h = handles_.at(i).value();
      CUmemGenericAllocationHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      DIPU_DRIVER_CHECK(cuMemUnmap(ptr_ + segment_size_ * i, segment_size_));
      DIPU_DRIVER_CHECK(cuMemRelease(h));
    }
    trimHandles();
  }

  void trimHandles() {
    while (!handles_.empty() && !handles_.back()) {
      handles_.pop_back();
    }
  }

  void forEachAllocatedRange(std::function<void(size_t, size_t)> fn) {
    size_t start = 0;
    for (auto i : c10::irange(handles_.size())) {
      if (handles_.at(i) && (i == 0 || !handles_.at(i - 1))) {
        start = i;
      }
      if (handles_.at(i) && (i + 1 == handles_.size() || !handles_.at(i + 1))) {
        fn(start, i + 1);
      }
    }
  }

  size_t numSegments(size_t size) const {
    return (size + segment_size_ - 1) / segment_size_;
  }

  size_t segmentLeft(const char* p) const {
    auto size = p - ptr();
    return size / segment_size_;
  }

  size_t segmentRight(const char* p) const {
    auto size = p - ptr();
    return numSegments(size);
  }

  SegmentRange rangeFromHandles(size_t begin, size_t end) const {
    return {ptr() + segment_size_ * begin, segment_size_ * (end - begin)};
  }

  int device_;
  deviceStream_t stream_;
  CUdeviceptr ptr_{};
  size_t max_handles_;
  size_t segment_size_;
  std::vector<c10::optional<CUmemGenericAllocationHandle>> handles_;
  // devices on which this memory should be mapped in addition
  // to the device where the physical memory lives (device_).
  std::vector<int> peers_;
};

ExpandableSegment* vendorCreateExpandableSegment(int device,
                                                 deviceStream_t stream,
                                                 size_t size,
                                                 std::vector<int> peers) {
  return new CUDAExpandableSegment(device, stream, size, peers);
}

}  // namespace dipu
