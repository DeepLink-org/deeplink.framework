// Copyright (c) 2024, DeepLink.
#include "DIPUTensorInferer.h"

#include <bitset>

#include <ATen/native/TypeProperties.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"

namespace dipu {

namespace internal {
DimVector compute_broadcast_shape(c10::IntArrayRef a, c10::IntArrayRef b) {
  // coumpute broadcast shape
  // for example: a = [2, 1, 3], b = [2, 1], the result shape would be [2, 2, 3]
  size_t ndim_a = a.size();
  size_t ndim_b = b.size();
  size_t ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
  // size of result is the bigger ndim
  DimVector result(ndim);

  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    // starting from the last index of a and b, then moving forward
    ptrdiff_t dim_a = ndim_a - ndim + i;
    ptrdiff_t dim_b = ndim_b - ndim + i;
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

struct ResultTypeState {
  at::ScalarType dimResult = at::ScalarType::Undefined;
  at::ScalarType wrappedResult = at::ScalarType::Undefined;
  at::ScalarType zeroResult = at::ScalarType::Undefined;
};

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

static inline at::ScalarType combine_categories(at::ScalarType higher,
                                                at::ScalarType lower) {
  if (c10::isComplexType(higher)) {
    return higher;
  } else if (c10::isComplexType(lower)) {
    // preserve value type of higher if it is floating type.
    if (c10::isFloatingType(higher)) {
      return c10::toComplexType(higher);
    }
    // in case of integral input
    // lower complex takes precedence.
    return lower;
  } else if (c10::isFloatingType(higher)) {
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

at::ScalarType result_type(const ResultTypeState& in_state) {
  return combine_categories(
      in_state.dimResult,
      combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

}  // namespace internal

void TensorInferer::compute_shape() {
  TORCH_CHECK(!inputs_.empty(),
              "No input tensors provided for shape computation");

  for (auto& t : inputs_) {
    auto shape = t.sizes();
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!shape.equals(shape_)) {
      shape_ = internal::compute_broadcast_shape(shape_, shape);
    }
  }
}

void TensorInferer::compute_dtype() {
  internal::ResultTypeState state = {};
  for (const auto& t : inputs_) {
    state = internal::update_result_type_state(t, state);
  }

  dtype_ = internal::result_type(state);
  TORCH_INTERNAL_ASSERT(dtype_ != at::ScalarType::Undefined);
}

at::Tensor TensorInferer::malloc_output() {
  TORCH_CHECK(dtype_ != at::ScalarType::Undefined,
              "Data type (dtype) for the tensor is undefined.")
  // Allocate the tensor with the given dtype and shape
  at::TensorOptions options = at::TensorOptions().dtype(dtype_).device(device_);
  return native::nodispatch::empty(shape_, options);
}

at::Tensor TensorInferer::infer_binary_op() {
  compute_shape();
  compute_dtype();
  return malloc_output();
}

at::Tensor TensorInferer::infer_unary_op() {
  // since `compute_shape` and `compute_dtype` are robust, we can reuse them
  return infer_binary_op();
}

at::Tensor TensorInferer::infer_comparison_op() {
  compute_shape();
  dtype_ = at::ScalarType::Bool;
  return malloc_output();
}

at::Tensor TensorInferer::infer_binary_float_op() {
  compute_shape();
  compute_dtype();
  // Promotes common dtype to the default float scalar type, if needed
  if (c10::isIntegralType(dtype_, /*includeBool=*/true)) {
    dtype_ = c10::typeMetaToScalarType(c10::get_default_dtype());
  }
  return malloc_output();
}

at::Tensor TensorInferer::infer_reduce_op(c10::OptionalIntArrayRef dim,
                                          bool keep_dim,
                                          c10::optional<at::ScalarType> dtype) {
  TORCH_CHECK(!inputs_.empty(), "Reduce op requires at least one input.");
  const auto& input_tensor = inputs_[0];
  int64_t ndim = input_tensor.dim();

  constexpr int64_t bitset_size = 64;
  std::bitset<bitset_size> dim_mask;
  if (!dim.has_value() || dim->empty()) {
    dim_mask.flip();  // All dimensions are reduced if `dim` is empty.
  } else {
    for (const auto& d : dim.value()) {
      TORCH_CHECK(d < ndim, "Dimension out of range.");
      TORCH_CHECK(!dim_mask[d], "Dimension ", d,
                  " appears multiple times in the list of dimensions.");
      dim_mask.set(d);
    }
  }

  shape_ = input_tensor.sizes();
  for (int64_t i = ndim - 1; i >= 0; --i) {
    if (dim_mask[i]) {
      if (keep_dim) {
        shape_[i] = 1;
      } else {
        shape_.erase(shape_.begin() + i);
      }
    }
  }

  compute_dtype();
  return malloc_output();
}

}  // namespace dipu
