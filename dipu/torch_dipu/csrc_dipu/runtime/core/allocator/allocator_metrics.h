#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include "csrc_dipu/metrics/export.h"

namespace dipu {

class AllocatorMetrics {
  using integer = metrics::ExportedInteger;

  std::mutex mutable mutex;
  std::unordered_map<void*, std::size_t> memory;

  metrics::LabeledIntegerCounter allocate_nullptr_count;
  metrics::LabeledIntegerCounter allocate_duplicated_count;
  metrics::LabeledIntegerCounter deallocate_nullptr_count;
  metrics::LabeledIntegerCounter deallocate_unexpected_count;

  metrics::LabeledIntegerHistogram allocate_size;
  metrics::LabeledIntegerHistogram deallocate_size;

 public:
  explicit AllocatorMetrics(
      metrics::Collector::labelset const& labels,
      metrics::Collector& co = metrics::default_collector())

      : allocate_nullptr_count{co.make_integer_counter("allocator_event_count",
                                                       "")
                                   .with(labels({{"method", "allocate"},
                                                 {"event", "nullptr"}}))},
        allocate_duplicated_count{allocate_nullptr_count.with(
            {{"method", "allocate"}, {"event", "duplicated"}})},
        deallocate_nullptr_count{allocate_nullptr_count.with(
            {{"method", "deallocate"}, {"event", "nullptr"}})},
        deallocate_unexpected_count{allocate_nullptr_count.with(
            {{"method", "deallocate"}, {"event", "unexpected"}})},

        allocate_size{co.make_integer_histogram("allocator_size", "", exp2())
                          .with(labels({{"method", "allocate"}}))},
        deallocate_size{allocate_size.with({{"method", "deallocate"}})}  //
  {}

  void allocate(void* data, std::size_t size) {
    if (size == 0 or not metrics::enable()) {
      // do nothing

    } else if (data == nullptr) {
      allocate_nullptr_count.inc();

    } else if (insert(data, size)) {
      allocate_size.put(static_cast<integer>(size));

    } else {
      allocate_duplicated_count.inc();
    }
  }

  void deallocate(void* data) {
    if (not metrics::enable()) {
      // do nothing

    } else if (data == nullptr) {
      deallocate_nullptr_count.inc();

    } else if (auto size = std::size_t(); remove(data, size)) {
      deallocate_size.put(static_cast<integer>(size));

    } else {
      deallocate_unexpected_count.inc();
    }
  }

  void set_device_number(std::string const& device) {
    std::scoped_lock _(mutex);
#define _OVERWRITE_DEVICE(field) field = (field).with({{"device", device}})
    _OVERWRITE_DEVICE(allocate_nullptr_count);
    _OVERWRITE_DEVICE(allocate_duplicated_count);
    _OVERWRITE_DEVICE(deallocate_nullptr_count);
    _OVERWRITE_DEVICE(deallocate_unexpected_count);
    _OVERWRITE_DEVICE(allocate_size);
    _OVERWRITE_DEVICE(deallocate_size);
#undef _OVERWRITE_DEVICE
  }

 private:
  auto insert(void* data, std::size_t size) -> bool {
    std::scoped_lock _(mutex);
    return memory.emplace(data, size).second;
  }

  auto remove(void* data, std::size_t& size) -> bool {
    std::scoped_lock _(mutex);
    if (auto iter = memory.find(data); iter != memory.end()) {
      size = iter->second;
      memory.erase(iter);
      return true;
    }
    return false;
  }

  auto static exp2(std::size_t from = 4U) -> std::vector<integer> {
    auto output = std::vector<integer>();
    output.reserve(std::numeric_limits<uint32_t>::digits);
    for (auto i = from; i < std::numeric_limits<uint32_t>::digits; ++i) {
      output.push_back(static_cast<integer>(1ULL << i));
    }
    return output;
  }
};

// TODO(refactor): replace hash table with a lazy array.
class GlobalAllocatorGroupMetrics {
  metrics::Collector::labelset labels;
  std::shared_mutex mutable mutex;
  std::unordered_map<c10::DeviceIndex, AllocatorMetrics> pool;

 private:
  explicit GlobalAllocatorGroupMetrics(metrics::Collector::labelset labels)
      : labels(std::move(labels)) {}

 public:
  auto inline static device_allocator_metrics()
      -> GlobalAllocatorGroupMetrics& {
    // Using * to avoid being destructed.
    auto static instance =
        new GlobalAllocatorGroupMetrics({{"type", "device"}});
    return *instance;
  }

  auto inline static host_allocator_metrics() -> GlobalAllocatorGroupMetrics& {
    // Using * to avoid being destructed.
    auto static instance = new GlobalAllocatorGroupMetrics({{"type", "host"}});
    return *instance;
  }

  auto operator[](c10::DeviceIndex index) -> AllocatorMetrics& {
    {
      std::shared_lock _(mutex);
      if (auto iter = pool.find(index); iter != pool.end()) {
        return iter->second;
      }
    }
    {
      auto& o = metrics::default_collector();
      auto ls = labels({{"device", std::to_string(index)}});
      std::unique_lock _(mutex);
      return pool.try_emplace(index, std::move(ls), o).first->second;
    }
  }
};

}  // namespace dipu
