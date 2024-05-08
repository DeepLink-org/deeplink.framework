// Copyright (c) 2023, DeepLink.
#include "OpUtils.hpp"

namespace dipu {
namespace native {

at::DimVector compute_broadcast_shape(c10::IntArrayRef a, c10::IntArrayRef b) {
  size_t ndim_a = a.size();
  size_t ndim_b = b.size();
  size_t ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
  // size of result is the bigger ndim
  at::DimVector result(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = static_cast<ptrdiff_t>(ndim) - 1; i >= 0; --i) {
    // starting from the last index of a and b, then moving forward
    ptrdiff_t dim_a =
        static_cast<ptrdiff_t>(ndim_a) - static_cast<ptrdiff_t>(ndim) + i;
    ptrdiff_t dim_b =
        static_cast<ptrdiff_t>(ndim_b) - static_cast<ptrdiff_t>(ndim) + i;
    // if the index is smaller than 0, consider it as 1
    auto size_a = (dim_a >= 0) ? a[dim_a] : 1;
    auto size_b = (dim_b >= 0) ? b[dim_b] : 1;

    TORCH_CHECK(size_a == size_b || size_a == 1 || size_b == 1,
                "The size of tensor a (", size_a,
                ") must match the size of tensor b (", size_b,
                ") at non-singleton dimension ", i);

    // 1 is mapped to the other size (even for 0).
    result[i] = size_a == 1 ? size_b : size_a;
  }

  return result;
}

}  // namespace native
}  // namespace dipu
