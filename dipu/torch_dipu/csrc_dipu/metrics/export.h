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

  metrics::labeled_integer_gauge allocate_size_total;
  metrics::labeled_integer_gauge deallocate_size_total;

  metrics::labeled_integer_histogram allocate_size_frequency;
  metrics::labeled_integer_histogram deallocate_size_frequency;

 public:
  explicit allocator_metrics(metrics::collector<char>& collector,
                             metrics::labelset<char> const& labels)
      : allocate_nullptr_count(
            collector.make_integer_counter("allocator_event_count", "TODO")
                .with(labels)
                .with({{"method", "allocate"}, {"event", "nullptr"}})),
        allocate_duplicated_count(allocate_nullptr_count.with(
            {{"method", "allocate"}, {"event", "duplicated"}})),
        deallocate_nullptr_count(allocate_nullptr_count.with(
            {{"method", "deallocate"}, {"event", "nullptr"}})),
        deallocate_unexpected_count(allocate_nullptr_count.with(
            {{"method", "deallocate"}, {"event", "unexpected"}})),

        allocate_size_total(
            collector.make_integer_gauge("allocator_size_total", "TODO")
                .with(labels)
                .with({{"method", "allocate"}})),
        deallocate_size_total(
            allocate_size_total.with({{"method", "deallocate"}})),

        allocate_size_frequency(
            collector
                .make_integer_histogram("allocator_size_frequency", "TODO",
                                        exp2())
                .with(labels)
                .with({{"method", "allocate"}})),
        deallocate_size_frequency(
            allocate_size_frequency.with({{"method", "deallocate"}}))
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
      allocate_size_total.add(static_cast<metrics::exported_integer>(size));
      allocate_size_frequency.put(static_cast<metrics::exported_integer>(size));
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

      deallocate_size_total.add(static_cast<metrics::exported_integer>(size));
      deallocate_size_frequency.put(
          static_cast<metrics::exported_integer>(size));
    }
  }

  void rename(std::string const& device) {
    std::scoped_lock _(mutex);

    allocate_nullptr_count = allocate_nullptr_count.with({{"device", device}});
    allocate_duplicated_count =
        allocate_duplicated_count.with({{"device", device}});
    deallocate_nullptr_count =
        deallocate_nullptr_count.with({{"device", device}});
    deallocate_unexpected_count =
        deallocate_unexpected_count.with({{"device", device}});
    allocate_size_total = allocate_size_total.with({{"device", device}});
    deallocate_size_total = deallocate_size_total.with({{"device", device}});
    allocate_size_frequency =
        allocate_size_frequency.with({{"device", device}});
    deallocate_size_frequency =
        deallocate_size_frequency.with({{"device", device}});
  }

 private:
  auto static exp2(std::size_t from = 4U)
      -> std::vector<metrics::exported_integer> {
    auto output = std::vector<metrics::exported_integer>();
    output.reserve(std::numeric_limits<uint32_t>::digits);
    for (auto i = from; i < std::numeric_limits<uint32_t>::digits; ++i) {
      output.push_back(1LL << i);
    }
    return output;
  }
};

auto extern default_metrics_collector() -> metrics::collector<char>&;
auto extern default_device_allocator_metrics_producer() -> allocator_metrics&;
auto extern default_host_allocator_metrics_producer() -> allocator_metrics&;

}  // namespace dipu
