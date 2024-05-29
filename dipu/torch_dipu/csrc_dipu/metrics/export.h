#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <string>

#include "collector.h"

namespace dipu {

class allocator_metrics {
 private:
  std::mutex mutable mutex;
  std::unordered_map<void*, std::size_t> memory;

  metrics::labeled_integer_counter allocate_nullptr_count;
  metrics::labeled_integer_counter allocate_duplicated_count;
  metrics::labeled_integer_counter deallocate_nullptr_count;
  metrics::labeled_integer_counter deallocate_unexpected_count;

  metrics::labeled_integer_histogram allocate_size;
  metrics::labeled_integer_histogram deallocate_size;

 public:
  explicit allocator_metrics(metrics::collector<char>& collector,
                             metrics::labelset<char> const& labels)
      : allocate_nullptr_count(
            collector.make_integer_counter("allocator_event_count", "TODO")
                .with(labels)
                .with({{"method", "allocate"}, {"event", "nullptr"}})),
        allocate_duplicated_count(  // Reuse allocate_nullptr_count as they are
                                    // in a same group.
            allocate_nullptr_count.with(
                {{"method", "allocate"}, {"event", "duplicated"}})),
        deallocate_nullptr_count(  //
            allocate_nullptr_count.with(
                {{"method", "deallocate"}, {"event", "nullptr"}})),
        deallocate_unexpected_count(  //
            allocate_nullptr_count.with(
                {{"method", "deallocate"}, {"event", "unexpected"}})),

        allocate_size(
            collector.make_integer_histogram("allocator_size", "TODO", exp2())
                .with(labels)
                .with({{"method", "allocate"}})),
        deallocate_size(allocate_size.with({{"method", "deallocate"}}))
  //
  {}

  void allocate(void* data, std::size_t size) {
    std::scoped_lock _(mutex);

    if (size == 0) {
      // do nothing
    } else if (data == nullptr) {
      allocate_nullptr_count.inc();

    } else if (not memory.emplace(data, size).second) {
      allocate_duplicated_count.inc();

    } else {
      allocate_size.put(static_cast<metrics::exported_integer>(size));
    }
  }

  void deallocate(void* data) {
    std::scoped_lock _(mutex);

    if (data == nullptr) {
      deallocate_nullptr_count.inc();

    } else if (auto iter = memory.find(data); iter == memory.end()) {
      deallocate_unexpected_count.inc();

    } else {
      auto size = iter->second;
      memory.erase(iter);

      deallocate_size.put(static_cast<metrics::exported_integer>(size));
    }
  }

  void set_device_number(std::string const& device) {
    std::scoped_lock _(mutex);
#define _OVERRIDE_DEVICE(field) field = (field).with({{"device", device}})
    _OVERRIDE_DEVICE(allocate_nullptr_count);
    _OVERRIDE_DEVICE(allocate_duplicated_count);
    _OVERRIDE_DEVICE(deallocate_nullptr_count);
    _OVERRIDE_DEVICE(deallocate_unexpected_count);
    _OVERRIDE_DEVICE(allocate_size);
    _OVERRIDE_DEVICE(deallocate_size);
#undef _OVERRIDE_DEVICE
  }

 private:
  auto static exp2(std::size_t from = 4U)
      -> std::vector<metrics::exported_integer> {
    auto output = std::vector<metrics::exported_integer>();
    output.reserve(std::numeric_limits<uint32_t>::digits);
    for (auto i = from; i < std::numeric_limits<uint32_t>::digits; ++i) {
      output.push_back(static_cast<metrics::exported_integer>(1ULL << i));
    }
    return output;
  }
};

auto extern default_metrics_collector() -> metrics::collector<char>&;
auto extern default_device_allocator_metrics_producer() -> allocator_metrics&;
auto extern default_host_allocator_metrics_producer() -> allocator_metrics&;

}  // namespace dipu
