#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <vector>

namespace dipu::metrics::detail {

// Is T one of U...
template <typename T, typename... U>
auto constexpr inline oneof = (std::is_same_v<T, U> || ...);

template <typename I>
struct scalar {
  static_assert(oneof<I, float, double, int64_t, uint64_t>);

  std::atomic<I> value;

  auto fetch_add(I number) -> I {
    // Require C++20 to fetch_add or fetch_sub floating point type.
    auto constexpr CPP17 = 201703L;
    if constexpr (std::is_integral_v<I> or __cplusplus > CPP17) {
      return value.fetch_add(number);

    } else {
      auto x = value.load(std::memory_order_relaxed);
      while (not value.compare_exchange_weak(x, x + number)) {
      }
      return x;
    }
  }

  auto fetch_sub(I number) -> I {
    // Require C++20 to fetch_add or fetch_sub floating point type.
    auto constexpr CPP17 = 201703L;
    if constexpr (std::is_integral_v<I> or __cplusplus > CPP17) {
      return value.fetch_sub(number);

    } else {
      auto x = value.load(std::memory_order_relaxed);
      while (not value.compare_exchange_weak(x, x - number)) {
      }
      return x;
    }
  }

  explicit scalar(I number) noexcept : value(number) {}
  ~scalar() = default;
  scalar() noexcept = default;
  scalar(scalar&& o) noexcept : value(o.value.load()) {}
  scalar(scalar const& o) noexcept : value(o.value.load()) {}
  scalar& operator=(scalar&& o) noexcept { value.store(o.value.load()); }
  scalar& operator=(scalar const& o) noexcept { value.store(o.load()); }
};

}  // namespace dipu::metrics::detail

namespace dipu::metrics {

// Prometheus style counter (or sum) value.
//
// Only increasing or reseting is allowed.
template <typename I>
class counter : protected detail::scalar<I> {
  using detail::scalar<I>::value;
  using detail::scalar<I>::fetch_add;

 public:
  using value_type = I;
  using detail::scalar<I>::scalar;

  auto get() const noexcept -> value_type { return value.load(); }
  auto inc() noexcept -> void { fetch_add(value_type{1}); }
  auto add(value_type number) noexcept -> void { fetch_add(number); }
  auto reset() noexcept -> void { value.store(value_type{0}); }
};

// Prometheus style gauge value.
template <typename I>
class gauge : protected detail::scalar<I> {
  using detail::scalar<I>::value;
  using detail::scalar<I>::fetch_add;
  using detail::scalar<I>::fetch_sub;

 public:
  using value_type = I;
  using detail::scalar<I>::scalar;

  auto get() const noexcept -> value_type { return value.load(); }
  auto inc() noexcept -> void { fetch_add(value_type{1}); }
  auto dec() noexcept -> void { fetch_sub(value_type{1}); }
  auto set(value_type number) noexcept -> void { value.store(number); }
  auto add(value_type number) noexcept -> void { fetch_add(number); }
  auto sub(value_type number) noexcept -> void { fetch_sub(number); }
  auto reset() noexcept -> void { value.store(value_type{0}); }
};

template <typename I>
class histogram {
  counter<I> summation;
  std::vector<I> thresholds;
  std::vector<counter<uint64_t>> buckets;

 public:
  using value_type = I;

  auto sum() const noexcept -> value_type { return summation.get(); }

  auto put(value_type number) noexcept -> void {
    summation.add(number);
    select_bucket(number).inc();
  }

  auto reset() noexcept -> void {
    // Warning: race condition may happen
    summation.reset();
    for (auto& bucket : buckets) {
      bucket.reset();
    }
  }

  auto get_thresholds() const noexcept -> std::vector<value_type> const& {
    return thresholds;
  }

  template <typename T>
  auto get_buckets() const -> std::vector<T> {
    auto out = std::vector<T>();
    out.reserve(buckets.size());
    for (auto& number : buckets) {
      out.push_back(number.get());
    }
    return out;
  }

  explicit histogram(std::vector<value_type> thresholds)
      : thresholds(monotonic(std::move(thresholds))),
        buckets(this->thresholds.size() + 1 /* +inf */) {}

  histogram(std::initializer_list<value_type> thresholds)
      : thresholds(monotonic(thresholds)),
        buckets(this->thresholds.size() + 1 /* +inf */) {}

 private:
  auto select_bucket(I number) -> counter<uint64_t>& {
    auto iter = std::lower_bound(thresholds.begin(), thresholds.end(), number);
    auto index = std::distance(thresholds.begin(), std::move(iter));
    return buckets[index];
  }

  template <typename T>
  auto monotonic(T&& list) -> std::vector<I> {
    auto out = static_cast<std::vector<I>>(std::forward<T>(list));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    out.shrink_to_fit();
    return out;
  }
};

template <typename I>
class summary {
  // unimplemented
 public:
  using value_type = I;
};

}  // namespace dipu::metrics

namespace dipu::metrics::detail {

// This wrapper struct has two tasks:
//
// 1. Provide access to private members.
// 2. Merge duplicated code from value_operation to here.
//
// Without this struct, we need to provide some same access methods and perform
// static_cast for every struct value_operation.
template <typename T>
struct value_access {
  decltype(auto) value() noexcept { return static_cast<T*>(this)->access(); }
  decltype(auto) value() const noexcept {
    return static_cast<T const*>(this)->access();
  }
};

// CRTP value operations for labeled_value
//
// Those struct are used to provide interface for labeled value.
template <typename T>
struct value_operation {};

template <template <typename, typename> class V, typename S, typename I>
struct value_operation<V<S, counter<I>>> : value_access<V<S, counter<I>>> {
  auto get() const noexcept -> I { return value().get(); }
  auto inc() noexcept -> void { value().inc(); }
  auto add(I number) noexcept -> void { value().add(number); }
  auto reset() noexcept -> void { value().reset(); }

 private:
  using value_access<V<S, counter<I>>>::value;
};

template <template <typename, typename> class V, typename S, typename I>
struct value_operation<V<S, gauge<I>>> : value_access<V<S, gauge<I>>> {
  auto get() const noexcept -> I { return value().get(); }
  auto inc() noexcept -> void { value().inc(); }
  auto dec() noexcept -> void { value().dec(); }
  auto set(I number) noexcept -> void { value().set(number); }
  auto add(I number) noexcept -> void { value().add(number); }
  auto sub(I number) noexcept -> void { value().sub(number); }
  auto reset() noexcept -> void { value().reset(); }

 private:
  using value_access<V<S, gauge<I>>>::value;
};

template <template <typename, typename> class V, typename S, typename I>
struct value_operation<V<S, histogram<I>>> : value_access<V<S, histogram<I>>> {
  template <typename T>
  [[nodiscard]] auto buckets() const -> std::vector<T> {
    return value().template get_buckets<T>();
  }
  auto thresholds() const noexcept -> std::vector<I> const& {
    return value().get_thresholds();
  }
  auto put(I number) noexcept -> void { value().put(number); }
  auto reset() noexcept -> void { value().reset(); }

 private:
  using value_access<V<S, histogram<I>>>::value;
};

}  // namespace dipu::metrics::detail
