#pragma once

#include <array>
#include <utility>

namespace dipu {

// Create an array of functor. Each functor should provide a 'value()' function
// to return a reference of a static variable. Thus the value could be lazily
// initialized.
template <typename T, std::size_t N>
struct make_static_function_array {
  template <typename U = std::make_index_sequence<N>>
  struct slot {};
  template <std::size_t... I>
  struct slot<std::index_sequence<I...>> {
    // deduction friendly
    using type = std::decay_t<decltype(T::template value<0>)>;
    // the actual array
    auto inline static constexpr value =
        std::array<type, sizeof...(I)>{T::template value<I>...};
  };
};

template <typename T, std::size_t N>
using static_function_array =
    typename make_static_function_array<T, N>::template slot<>;

}  // namespace dipu
