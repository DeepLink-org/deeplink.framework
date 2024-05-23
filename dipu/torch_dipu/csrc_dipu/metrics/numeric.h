#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>

namespace dipu::metrics::detail {

// Define the min, max, zero value of some types.
template <typename T, typename = void>
struct limits {};
template <typename T>
using limits_t = typename limits<T>::type;
template <typename T>
struct limits<T, std::enable_if_t<std::numeric_limits<T>::is_specialized>> {
  using type = T;
  static constexpr type max() noexcept { return std::numeric_limits<T>::max(); }
  static constexpr type min() noexcept { return std::numeric_limits<T>::max(); }
  static constexpr type zero() noexcept { return T{}; }
};
template <typename T, typename U>
struct limits<std::chrono::duration<T, U>, void> {
  using type = std::chrono::duration<T, U>;
  static constexpr type max() noexcept { return type::max(); }
  static constexpr type min() noexcept { return type::max(); }
  static constexpr type zero() noexcept { return type::zero(); }
};

// Extend bit size of number type.
// - Map integers {int, bool, char ...} to int64_t, and {float} to double.
// - Otherwise, do nothing.
template <typename T, typename = void>
struct saturate_widen {
  using type = T;
};
template <typename T>
struct saturate_widen<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
};
template <typename T>
struct saturate_widen<T, std::enable_if_t<std::is_floating_point_v<T>>> {
  using type = double;
};
template <typename T>
using saturate_widen_t = typename saturate_widen<T>::type;

// Perform a safe "++x".
template <typename T>
auto constexpr saturate_increase(T& value) noexcept
    -> std::void_t<limits_t<T>> {
  if (value != limits<T>::max()) {  // likely
    ++value;
  }
}

// Perform a safe "+=".
template <typename T>
auto constexpr saturate_increase(T& value, T delta) noexcept
    -> std::void_t<limits_t<T>> {
  auto limit = delta < limits<T>::zero() ? limits<T>::min() : limits<T>::max();
  if (value == limit) {  // unlikely
    // do nothing
  } else if (limit - delta <= value) {  // unlikely
    value = limit;
  } else {
    value += delta;
  }
}

}  // namespace dipu::metrics::detail

namespace dipu::metrics {

template <typename I>
struct maximum_minimum_summation {
  static_assert(std::is_trivial_v<I>);
  using value_type = I;
  using widen_type = detail::saturate_widen_t<I>;
  using count_type = std::size_t;

  count_type count{};
  widen_type summation{};
  value_type minimum{};
  value_type maximum{};

  void constexpr put(value_type input) noexcept {
    if (count > 0) {
      if (maximum < input) {
        maximum = input;
      }
      if (minimum > input) {
        minimum = input;
      }
    } else {
      maximum = input;
      minimum = input;
    }

    detail::saturate_increase(summation, static_cast<widen_type>(input));
    detail::saturate_increase(count);
  }

  void constexpr clear() noexcept {
    count = {};
    summation = {};
    minimum = {};
    maximum = {};
  }
};

template <typename I, I Base = 2>
struct fixed_size_exponential_histogram {
 private:
  auto constexpr static log(I number) noexcept -> std::size_t {
    auto i = 0;
    while (Base <= number) {
      number /= Base;
      ++i;
    }
    return i;
  }

 public:
  static_assert(std::is_trivial_v<I> and Base > 1);
  using value_type = I;
  using array_type =
      std::array<value_type, 1U + log(detail::limits<value_type>::max())>;

  array_type value{};

  void put(value_type input) noexcept {
    if (auto index = log(input); index < value.size()) {
      ++value[index];
    }
  }

  void pop(value_type input) noexcept {
    if (auto index = log(input); index < value.size() and 0 < value[index]) {
      --value[index];
    }
  }

  void clear() noexcept { std::memset(value.data(), 0, sizeof(array_type)); }
};

template <typename I, std::size_t N>
struct fixed_size_samples {
  static_assert(std::is_trivial_v<I>);
  using value_type = I;
  using count_type = std::size_t;
  using array_type = std::array<value_type, N>;

  count_type count{};
  array_type value{};

  void put(value_type input) {
    using uniform = std::uniform_int_distribution<count_type>;
    auto static thread_local engine = std::mt19937{std::random_device()()};

    if (count < value.size()) {
      value[count] = input;
    } else if (auto index = uniform(0, count)(engine); index < value.size()) {
      value[index] = input;
    }

    detail::saturate_increase(count);
  }

  [[nodiscard]] bool constexpr is_sampled() const noexcept {
    return value.size() != count;
  };

  [[nodiscard]] double constexpr sample_ratio() const noexcept {
    return static_cast<double>(value.size()) / static_cast<double>(count);
  };

  void constexpr clear() noexcept { count = 0; }
};

}  // namespace dipu::metrics
