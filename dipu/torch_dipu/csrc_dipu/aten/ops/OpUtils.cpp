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

static inline at::ScalarType promote_skip_undefined(at::ScalarType a,
                                                    at::ScalarType b) {
  if (a == at::ScalarType::Undefined) {
    return b;
  }
  if (b == at::ScalarType::Undefined) {
    return a;
  }
  return c10::promoteTypes(a, b);
}


static inline at::ScalarType combine_categories(at::ScalarType higher,
                                                at::ScalarType lower) {
  if (c10::isComplexType(higher)) {
    return higher;
  }
  if (c10::isComplexType(lower)) {
    // preserve value type of higher if it is floating type.
    if (c10::isFloatingType(higher)) {
      return c10::toComplexType(higher);
    }
    // in case of integral input
    // lower complex takes precedence.
    return lower;
  }
  if (c10::isFloatingType(higher)) {
    return higher;
  }
  if (higher == at::ScalarType::Bool || c10::isFloatingType(lower)) {
    return promote_skip_undefined(higher, lower);
  }
  if (higher != at::ScalarType::Undefined) {
    return higher;
  }
  return lower;
}

ResultTypeState update_result_type_state(const at::Tensor& tensor,
                                         const ResultTypeState& in_state) {
  ResultTypeState new_state = in_state;
  at::ScalarType current = tensor.scalar_type();
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    if (c10::isComplexType(current)) {
      current = c10::typeMetaToScalarType(at::get_default_complex_dtype());
    } else if (c10::isFloatingType(current)) {
      current = c10::typeMetaToScalarType(at::get_default_dtype());
    }
  }
  if (tensor.dim() > 0) {
    new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
  } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    new_state.wrappedResult =
        promote_skip_undefined(in_state.wrappedResult, current);
  } else {
    new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
  }
  return new_state;
}

ResultTypeState update_result_type_state(const at::Scalar& scalar,
                                         const ResultTypeState& in_state) {
  ResultTypeState new_state = in_state;
  at::ScalarType current = scalar.type();
  if (c10::isComplexType(current)) {
    current = c10::typeMetaToScalarType(at::get_default_complex_dtype());
  } else if (c10::isFloatingType(current)) {
    current = c10::typeMetaToScalarType(at::get_default_dtype());
  }
  new_state.wrappedResult =
      promote_skip_undefined(in_state.wrappedResult, current);
  return new_state;
}

at::ScalarType result_type(const ResultTypeState& in_state) {
  return combine_categories(
      in_state.dimResult,
      combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

at::ScalarType result_type(const at::Tensor& tensor, const at::Tensor& other) {
  ResultTypeState state = {};
  state = update_result_type_state(tensor, state);
  state = update_result_type_state(other, state);
  return result_type(state);
}

at::ScalarType result_type(const at::Tensor& tensor, const at::Scalar& other) {
  ResultTypeState state = {};
  state = update_result_type_state(tensor, state);
  state = update_result_type_state(other, state);
  return result_type(state);
}

}  // namespace native
}  // namespace dipu
