// Copyright (c) 2024, DeepLink.

#include "csrc_dipu/runtime/core/allocator/ExpandableSegment.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include "csrc_dipu/vendor/ascend/basecommimpl.hpp"

namespace dipu {

class AscendExpandableSegment : public ExpandableSegment {
 public:
  AscendExpandableSegment(int device, deviceStream_t stream, size_t size)
      : device_(device),
        stream_(stream),
        // 2MB for small pool, 20MB for large pool
        segment_size_(size) {
    devapis::DIPUDeviceProperties prop = devproxy::getDeviceProperties(device_);
    // we allocate enough address space for 1 1/8 the total memory on the GPU.
    // This allows for some cases where we have to unmap pages earlier in the
    // segment to put them at the end.
    max_handles_ = numSegments(prop.totalGlobalMem + prop.totalGlobalMem / 8);
    DIPU_CALLACLRT(aclrtReserveMemAddress(&ptr_, segment_size_ * max_handles_,
                                          0ULL, nullptr, 1ULL));
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
      aclrtDrvMemHandle handle = nullptr;
      aclrtPhysicalMemProp prop = {};
      prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
      prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
      prop.memAttr = ACL_HBM_MEM_HUGE;
      prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
      prop.reserve = 0;
      auto status = aclrtMallocPhysical(&handle, segment_size_, &prop, 0);
      if (status == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        for (auto j : c10::irange(begin, i)) {
          auto h = handles_.at(j).value();
          handles_.at(j) = c10::nullopt;
          DIPU_CALLACLRT(aclrtFreePhysical(h));
        }
        trimHandles();
        return rangeFromHandles(begin, begin);
      }
      DIPU_CALLACLRT(status);
      handles_.at(i) = handle;
    }

    for (auto i : c10::irange(begin, end)) {
      DIPU_CALLACLRT(aclrtMapMem(ptr_ + i * segment_size_, segment_size_, 0,
                                 handles_.at(i).value(), 0ULL));
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

  char* ptr() const override { return static_cast<char*>(ptr_); }
  size_t size() const override { return max_handles_ * segment_size_; }

 public:
  ~AscendExpandableSegment() noexcept override {
    forEachAllocatedRange(
        [&](size_t begin, size_t end) { unmapHandles(begin, end); });
    DIPU_CALLACLRT(aclrtReleaseMemAddress(ptr_));
  }

 private:
  void unmapHandles(size_t begin, size_t end) {
    // note: unlike cudaFree, MemUnmap and MemRelease do
    // not appear to synchronize in all cases, so we have to wait for the
    // stream to finish before this memory is truly free.

    // cannot call c10::cuda::stream_synchronize because
    // it might grab the GIL which can lead to a deadlock
    // Locking order must be GIL -> Allocator Lock
    devproxy::syncStream(stream_);
    for (auto i : c10::irange(begin, end)) {
      aclrtDrvMemHandle h = handles_.at(i).value();
      handles_.at(i) = c10::nullopt;
      DIPU_CALLACLRT(aclrtUnmapMem(ptr_ + segment_size_ * i));
      DIPU_CALLACLRT(aclrtFreePhysical(h));
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
  void* ptr_{};
  size_t max_handles_{};
  size_t segment_size_;
  std::vector<c10::optional<aclrtDrvMemHandle>> handles_;
};

ExpandableSegment* vendorCreateExpandableSegment(int device,
                                                 deviceStream_t stream,
                                                 size_t size) {
  return new AscendExpandableSegment(device, stream, size);
}

}  // namespace dipu
