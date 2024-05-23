#pragma once

#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "collector.h"
#include "numeric.h"
#include "statistics.h"
#include "utility.h"

namespace dipu::metrics {

inline constexpr bool enable_metrics_collection = true;

namespace detail {

struct with_mutex {
  std::mutex mutable mutex;
  [[nodiscard]] auto maybe_scoped_lock() const -> std::scoped_lock<std::mutex> {
    return std::scoped_lock(mutex);
  }
};

struct without_mutex {
  struct nothing {};
  auto static maybe_scoped_lock() noexcept -> nothing { return {}; }
};

}  // namespace detail

template <bool enable, bool with_lock>
class allocator_metrics_source
    : protected metrics_source,
      std::conditional_t<with_lock, detail::with_mutex, detail::without_mutex> {
 private:
  using remove_source = void (metrics_collector::*)(metrics_source&) noexcept;

  std::string key;
  std::unordered_map<void*, std::size_t> memory;
  detail::double_buffered<allocator_statistics> values;
  detail::defer<remove_source, metrics_collector&, metrics_source&> cleaner;

 protected:
  void before_update() override { values.flip(); }
  void update(statistics& s) override { s[key] = *values.consumer(); }

 public:
  explicit allocator_metrics_source(  //
      std::string unique_name, metrics_collector& collector)
      : key(std::move(unique_name)),
        cleaner(&metrics_collector::remove_source, collector, *this) {
    collector.insert_source(*this);
  }

  void allocate(void* data, std::size_t size) {
    auto _ = this->maybe_scoped_lock();

    if (size == 0) {
      // do nothing
    } else if (data == nullptr) {
      auto o = values.producer();
      o->allocate_nullptr_count += 1;

    } else if (not memory.emplace(data, size).second) {
      auto o = values.producer();
      o->allocate_duplicated_count += 1;

    } else {
      auto o = values.producer();
      o->used_size_summation += size;
      o->used_size_frequency.put(size);
      o->allocate_size.put(size);
      o->allocate_size_frequency.put(size);
    }
  }

  void deallocate(void* data) {
    auto _ = this->maybe_scoped_lock();

    if (data == nullptr) {
      auto o = values.producer();
      o->deallocate_nullptr_count += 1;

    } else if (auto iter = memory.find(data); iter == memory.end()) {
      auto o = values.producer();
      o->deallocate_unexpected_count += 1;

    } else {
      auto size = iter->second;
      memory.erase(iter);

      auto o = values.producer();
      o->used_size_summation -= size;
      o->used_size_frequency.pop(size);
      o->deallocate_size.put(size);
      o->deallocate_size_frequency.put(size);
    }
  }

  void rename(std::string const& unique_name) {
    auto _ = this->maybe_scoped_lock();
    key = unique_name;
  }
};

template <bool with_lock>
class allocator_metrics_source<false, with_lock> {
 public:
  explicit allocator_metrics_source(std::string const& /*unused*/,
                                    metrics_collector& /*unused*/) {}
  void allocate(void* data, std::size_t size) noexcept {}
  void deallocate(void* data) noexcept {}
};

template <bool use_lock>
using allocator_metrics_producer =
    allocator_metrics_source<enable_metrics_collection, use_lock>;

}  // namespace dipu::metrics
