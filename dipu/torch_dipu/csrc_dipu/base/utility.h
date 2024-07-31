#pragma once

#include <array>
#include <utility>

namespace dipu {

template <typename T, typename S>
struct make_static_value_array {};

template <typename T, std::size_t... I>
struct make_static_value_array<T, std::index_sequence<I...>> {
  using value_type = std::decay_t<decltype(T::template value<0>)>;
  using type = std::array<value_type, sizeof...(I)>;
  type static inline constexpr value{T::template value<I>...};
};

// Create an array of static values. Each type T should provide a static
// 'value()' function to return a reference of the underlying static variable.
//
// In this way, value can be lazily initialized.
template <typename T, std::size_t N>
// Note: in C++17 "inline constexpr" (without "static") is necessary to make
// sure there is only one entity.
auto inline constexpr make_static_value_array_v =
    make_static_value_array<T, std::make_index_sequence<N>>::value;

// A CRTP helper to simplify access method of static value holder.
//
// Usage:
//
// ```
// struct integer : static_value_array<foo, 16> {
//   template <std::size_t I> auto static value() -> int & {
//     auto static instance = 0;
//     return instance;
//   }
// };
//
// Then we can use `integer::get(1)` to access `integer::value<1>()`.
template <typename T, std::size_t N>
struct static_value_array {
  auto static array() -> decltype(auto) {
    auto static constexpr instance = make_static_value_array_v<T, N>;
    return instance;
  }
  template <typename... A>
  auto static get(std::size_t index, A&&... args) -> decltype(auto) {
    return array()[index](std::forward<A>(args)...);
  }
};

}  // namespace dipu
