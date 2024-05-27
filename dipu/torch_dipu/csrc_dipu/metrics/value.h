#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace dipu::metrics::detail {

template <typename T, typename... U>
auto constexpr inline oneof = (std::is_same_v<T, U> || ...);

template <typename T>
auto constexpr inline allowed_metrics_scalar_type =
    oneof<T, float, double, uint64_t, int64_t>;

}  // namespace dipu::metrics::detail

namespace dipu::metrics {

template <typename I>
class counter {
  static_assert(detail::allowed_metrics_scalar_type<I> and
                not std::is_floating_point_v<I>);

  std::atomic<I> value;

 public:
  using value_type = I;

  auto get() const noexcept -> value_type { return value.load(); }
  auto rst() noexcept -> void { value.store(static_cast<value_type>(0)); }
  auto inc() noexcept -> void { value.fetch_add(static_cast<value_type>(1)); }
  auto add(value_type number) noexcept -> void { value.fetch_add(number); }

  explicit counter(value_type number) noexcept : value(number) {}
  ~counter() = default;
  counter() noexcept = default;
  counter(counter&& other) noexcept : value(other.value.load()) {}
  counter(counter const& other) noexcept : value(other.value.load()) {}
  counter& operator=(counter&& other) noexcept {
    value.store(other.value.load());
  }
  counter& operator=(counter const& other) noexcept {
    value.store(other.load());
  }
};

template <typename I>
class gauge {
  static_assert(detail::allowed_metrics_scalar_type<I>);

  std::atomic<I> value;

 public:
  using value_type = I;

  // Note: require C++20 to fetch_add or fetch_sub floating point type.

  auto get() const noexcept -> value_type { return value.load(); }
  auto rst() noexcept -> void { value.store(static_cast<value_type>(0)); }
  auto inc() noexcept -> void { value.fetch_add(static_cast<value_type>(1)); }
  auto dec() noexcept -> void { value.fetch_sub(static_cast<value_type>(1)); }
  auto set(value_type number) noexcept -> void { value.store(number); }
  auto add(value_type number) noexcept -> void { value.fetch_add(number); }
  auto sub(value_type number) noexcept -> void { value.fetch_sub(number); }

  explicit gauge(value_type number) noexcept : value(number) {}
  ~gauge() = default;
  gauge() noexcept = default;
  gauge(gauge&& other) noexcept : value(other.value.load()) {}
  gauge(gauge const& other) noexcept : value(other.value.load()) {}
  gauge& operator=(gauge&& other) noexcept { value.store(other.value.load()); }
  gauge& operator=(gauge const& other) noexcept {
    value.store(other.value.load());
  }
};

template <typename I>
class histogram {
  static_assert(detail::allowed_metrics_scalar_type<I>);

  std::vector<I> thresholds;
  std::vector<counter<uint64_t>> buckets;

 public:
  using value_type = I;

  auto put(value_type number) noexcept -> void { select_bucket(number).inc(); }

  auto rst() noexcept -> void {
    for (auto& bucket : buckets) {
      // Race condition may happen
      bucket.rst();
    }
  }

  auto get_thresholds() const noexcept -> std::vector<value_type> const& {
    return thresholds;
  }

  template <typename T>
  auto get_buckets() const -> std::vector<T> {
    // ref:
    // https://github.com/OpenObservability/OpenMetrics/blob/main/specification/OpenMetrics.md#histogram
    auto size = buckets.size();
    auto sum = T{};
    auto out = std::vector<T>(size);
    for (auto i = std::size_t{}; i < size; ++i) {
      out[i] = sum += static_cast<T>(buckets[i].get());
    }
    return out;
  }

  explicit histogram(std::vector<value_type> thresholds)
      : thresholds(monotonic(std::move(thresholds))),
        buckets(this->thresholds.size() + 1 /* +inf */) {}

  histogram(std::initializer_list<value_type> thresholds)
      : thresholds(monotonic(thresholds)),
        buckets(this->thresholds.size() + 1 /* +inf */) {}

  histogram(value_type start, value_type delta, std::size_t number)
      : thresholds(number), buckets(number + 1 /* +inf */) {
    auto value = start;
    for (auto& threshold : thresholds) {
      threshold = value;
      value += delta;
    }
  }

  // ~histogram() = default;
  // histogram(histogram &&other) noexcept = default;
  // histogram(histogram const &other) = default;
  // histogram &operator=(histogram &&other) noexcept = default;
  // histogram &operator=(histogram const &other) = default;

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
  static_assert(detail::allowed_metrics_scalar_type<I>);

 public:
  using value_type = I;
  auto put(value_type number) noexcept -> void;
};

}  // namespace dipu::metrics

namespace dipu::metrics::detail {

/// get name by type
template <typename T, typename C>
struct value_name {};
template <typename I>
struct value_name<counter<I>, char> {
  auto static inline const name = "counter";
};
template <typename I>
struct value_name<gauge<I>, char> {
  auto static inline const name = "gauge";
};
template <typename I>
struct value_name<histogram<I>, char> {
  auto static inline const name = "histogram";
};

/// access value from derived type
template <typename T>
struct value_access {
  decltype(auto) value() noexcept { return static_cast<T*>(this)->access(); }
  decltype(auto) value() const noexcept {
    return static_cast<T const*>(this)->access();
  }
};

/// CRTP value operations for labeled_value, ignore type C
template <typename T>
struct value_operation {};

template <template <typename, typename> class V, typename I, typename C>
struct value_operation<V<counter<I>, C>> : value_access<V<counter<I>, C>> {
 private:
  using value_access<V<counter<I>, C>>::value;

 public:
  auto get() const noexcept -> I { return value().get(); }
  auto rst() noexcept -> void { value().rst(); }
  auto inc() noexcept -> void { value().inc(); }
  auto add(I number) noexcept -> void { value().add(number); }
};

template <template <typename, typename> class V, typename I, typename C>
struct value_operation<V<gauge<I>, C>> : value_access<V<gauge<I>, C>> {
 private:
  using value_access<V<gauge<I>, C>>::value;

 public:
  auto get() const noexcept -> I { return value().get(); }
  auto rst() noexcept -> void { value().rst(); }
  auto inc() noexcept -> void { value().inc(); }
  auto dec() noexcept -> void { value().dec(); }
  auto set(I number) noexcept -> void { value().set(number); }
  auto add(I number) noexcept -> void { value().add(number); }
  auto sub(I number) noexcept -> void { value().sub(number); }
};

template <template <typename, typename> class V, typename I, typename C>
struct value_operation<V<histogram<I>, C>> : value_access<V<histogram<I>, C>> {
 private:
  using value_access<V<histogram<I>, C>>::value;

 public:
  auto rst() noexcept -> void { value().rst(); }
  auto put(I number) noexcept -> void { value().put(number); }
  template <typename T>
  [[nodiscard]] auto buckets() const -> std::vector<T> {
    return value().template get_buckets<T>();
  }
  auto thresholds() const noexcept -> std::vector<I> const& {
    return value().get_thresholds();
  }
};

}  // namespace dipu::metrics::detail
