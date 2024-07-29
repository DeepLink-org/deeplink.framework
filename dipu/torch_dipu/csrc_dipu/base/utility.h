#pragma once

#include <array>
#include <utility>

namespace dipu {

template <typename T, std::size_t N, typename = std::make_index_sequence<N>>
struct make_static_value_array {};

template <typename T, std::size_t N, std::size_t... I>
struct make_static_value_array<T, N, std::index_sequence<I...>> {
  // deduction friendly.
  using value_type = std::decay_t<decltype(T::template value<0>)>;
  auto inline constexpr static value =
      std::array<value_type, N>{T::template value<I>...};
};

// Create an array of static values. Each type T should provide a static
// 'value()' function to return a reference of the underlying static variable.
//
// e.g. T is something like:
//
// ```
// struct foo {
//   template <std::size_t I> auto static value() -> int & {
//     auto static instance = 0;
//     return instance;
//   }
// };
// ```
//
// In this way, value can be lazily initialized.
template <typename T, std::size_t N>
// Note: in C++17 "inline constexpr" is necessary to make sure there is only one
// entity.
auto inline constexpr static_value_array = make_static_value_array<T, N>::value;

}  // namespace dipu
