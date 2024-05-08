// Copyright (c) 2024, DeepLink.
#include "DIPUOpInferrer.h"

#include <bitset>

#include <ATen/native/TypeProperties.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/aten/ops/OpUtils.hpp"

namespace dipu {

void OpInferrer::compute_shape() {
  TORCH_CHECK(!inputs_.empty(),
              "No input tensors provided for shape computation");

  for (const auto i : c10::irange(ntensors())) {
    auto shape = tensor(i).sizes();
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!shape.equals(shape_)) {
      all_same_shape_ = false;
      shape_ = native::compute_broadcast_shape(shape_, shape);
    }
  }
}

void OpInferrer::add_input(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.defined(), "Input tensor is undefined");
  inputs_.push_back(c10::MaybeOwned<at::Tensor>::borrowed(tensor));
}

at::Tensor OpInferrer::malloc_output() {
  at::TensorOptions options =
      at::TensorOptions().dtype(dtype_).device(dipu::DIPU_DEVICE_TYPE);
  auto out = native::nodispatch::empty(shape_, options, memory_format_);

  if (!strides_.empty()) {
    out.as_strided_(shape_, strides_);
  }
  return out;
}

void OpInferrer::compute_dtype() {
  native::ResultTypeState state = {};
  for (const auto i : c10::irange(ntensors())) {
    state = native::update_result_type_state(tensor(i), state);
  }

  dtype_ = native::result_type(state);
  TORCH_INTERNAL_ASSERT(dtype_ != at::ScalarType::Undefined);
}

bool OpInferrer::fast_compute_memory_format() {
  if (!all_same_shape_) {
    return false;
  }

  bool is_contiguous = true;
  bool is_channels_last = true;
  bool is_non_overlapping_and_dense = true;

  // Iterate through input tensors to check their properties
  for (const auto i : c10::irange(ntensors())) {
    auto& t = tensor(i);
    if (t.defined()) {
      is_contiguous &= t.is_contiguous(at::MemoryFormat::Contiguous);
      is_channels_last &= t.is_contiguous(at::MemoryFormat::ChannelsLast);
      is_non_overlapping_and_dense &= t.is_non_overlapping_and_dense();
    }
  }

  // Decide the memory format based on properties
  if (is_contiguous) {
    memory_format_ = at::MemoryFormat::Contiguous;
    return true;
  }
  if (is_channels_last) {
    memory_format_ = at::MemoryFormat::ChannelsLast;
    return true;
  }
  if (is_non_overlapping_and_dense) {
    // Additional check for matching strides
    const auto& reference_strides = tensor(0).strides();
    for (const auto i : c10::irange(1, ntensors())) {
      if (!reference_strides.equals(tensor(i).strides())) {
        return false;
      }
    }
    // Use memory format of the first input
    memory_format_ = tensor(0).suggest_memory_format();
    return true;
  }

  return false;
}

std::vector<c10::DimVector> OpInferrer::compute_effective_strides() {
  std::vector<c10::DimVector> strides(ntensors(), c10::DimVector(ndim(), 0));
  for (int i = 0; i < ntensors(); ++i) {
    auto& t = tensor(i);
    auto original_shape = t.sizes();
    auto original_stride = t.strides();
    auto offset = ndim() - original_shape.size();

    for (int j = 0; j < original_shape.size(); ++j) {
      if (!(original_shape[j] == 1 && shape_[offset + j] != 1)) {
        strides[i][offset + j] = original_stride[j];
      }
    }
  }
  return strides;
}

// Calculate perm_ to sort the dimensions based on strides in ascending order.
// Then we can use the perm_ to calculate the output stride
void OpInferrer::compute_perm() {
  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  auto strides = compute_effective_strides();

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto i : c10::irange(ntensors())) {
      int64_t stride0 = strides[i][dim0];
      int64_t stride1 = strides[i][dim1];
      if (stride0 == 0 || stride1 == 0) {
        // move on to the next input if one of the dimensions is broadcasted
        continue;
      }
      if (stride0 < stride1) {
        return -1;
      }
      if (stride0 > stride1) {
        return 1;
      }
      // equal strides, use dimensions themselves as the tie-breaker.
      if (shape_[dim0] > shape_[dim1]) {
        return 1;
      }
    }
    return 0;
  };

  // insertion sort with support for ambiguous comparisons
  for (const auto i : c10::irange(1, ndim())) {
    size_t dim1 = i;
    // dim0 >= 0; dim0-- causes overflow
    for (size_t dim0 = i; dim0-- > 0;) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
}

void OpInferrer::compute_memory_format() {
  if (fast_compute_memory_format()) {
    return;
  }
  compute_perm();

  // Calculate strides based on perm_
  auto strides = c10::DimVector();
  int64_t next_stride = 1;
  for (const auto dim : c10::irange(ndim())) {
    strides.push_back(next_stride);
    next_stride *= shape_[perm_[dim]];
  }

  // calculate the final strides_
  strides_.resize(strides.size());
  for (const auto dim : c10::irange(ndim())) {
    strides_[perm_[dim]] = strides[dim];
  }
}

at::Tensor BinaryOpInferrer::infer_out(const at::Tensor& self,
                                       const at::Tensor& other) {
  add_input(self);
  add_input(other);
  compute_shape();
  compute_dtype();
  compute_memory_format();
  return malloc_output();
}

at::Tensor BinaryFloatOpInferrer::infer_out(const at::Tensor& self,
                                            const at::Tensor& other) {
  add_input(self);
  add_input(other);
  compute_shape();
  compute_dtype();
  // Promotes common dtype to the default float scalar type, if needed
  if (c10::isIntegralType(dtype_, /*includeBool=*/true)) {
    dtype_ = c10::typeMetaToScalarType(c10::get_default_dtype());
  }
  compute_memory_format();
  return malloc_output();
}

at::Tensor UnaryOpInferrer::infer_out(const at::Tensor& self) {
  add_input(self);
  compute_shape();
  compute_dtype();
  compute_memory_format();
  return malloc_output();
}

at::Tensor LogicOpInferrer::infer_out(const at::Tensor& self,
                                      const at::Tensor& other) {
  add_input(self);
  add_input(other);
  compute_shape();
  dtype_ = at::ScalarType::Bool;
  compute_memory_format();
  return malloc_output();
}

void ReduceOpInferrer::compute_shape(c10::OptionalIntArrayRef dim,
                                     bool keep_dim) {
  const auto& input_tensor = tensor(0);
  int64_t ndim = input_tensor.dim();

  constexpr int64_t bitset_size = 64;
  std::bitset<bitset_size> dim_mask;
  if (!dim.has_value() || dim->empty()) {
    dim_mask.flip();  // All dimensions are reduced if `dim` is empty.
  } else {
    for (const auto& d : dim.value()) {
      auto idx = c10::maybe_wrap_dim(d, ndim);
      TORCH_CHECK(idx >= 0 && idx < ndim, "Dimension out of range.");
      TORCH_CHECK(!dim_mask[idx], "Dimension ", idx,
                  " appears multiple times in the list of dimensions.");
      dim_mask.set(idx);
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
}

void ReduceOpInferrer::compute_dtype() {
  dtype_ = tensor(0).scalar_type();
  if (at::isIntegralType(dtype_, /*includeBool=*/true)) {
    dtype_ = c10::kLong;
  }
}

at::Tensor ReduceOpInferrer::infer_out(const at::Tensor& self,
                                       c10::OptionalIntArrayRef dim,
                                       bool keep_dim,
                                       c10::optional<at::ScalarType> dtype) {
  add_input(self);
  compute_shape(dim, keep_dim);
  if (dtype.has_value()) {
    dtype_ = dtype.value();
  } else {
    compute_dtype();
  }
  memory_format_ = at::MemoryFormat::Contiguous;
  return malloc_output();
}

// Check to see if the shape of tensors is compatible
// for being concatenated along a given dimension.
void CatOpInferrer::check_cat_shape_except_dim(size_t index, size_t index_2,
                                               int64_t dimension) {
  auto& first = tensor(index);
  auto& second = tensor(index_2);
  int64_t first_dims = first.dim();
  int64_t second_dims = second.dim();
  TORCH_CHECK(first_dims == second_dims,
              "Tensors must have same number of dimensions: got ", first_dims,
              " and ", second_dims);
  for (const auto dim : c10::irange(first_dims)) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = first.sizes()[dim];
    int64_t second_dim_size = second.sizes()[dim];
    TORCH_CHECK(first_dim_size == second_dim_size,
                "Sizes of tensors must match except in dimension ", dimension,
                ". Expected size ", static_cast<long long>(first_dim_size),
                " but got size ", static_cast<long long>(second_dim_size),
                " for tensor number ", index_2, " in the list.");
  }
}

void CatOpInferrer::compute_memory_format() {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (const auto i : c10::irange(ntensors())) {
    auto f = tensor(i).suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
      memory_format_ = f;
      return;
    }
    if (format.has_value() && format.value() != f) {
      memory_format_ = c10::MemoryFormat::Contiguous;
      return;
    }
    format = f;
  }
  memory_format_ = format.value();
}

void CatOpInferrer::compute_shape(int64_t dim) {
  // Look for the first valid tensor.
  size_t valid = ntensors();
  for (const auto i : c10::irange(ntensors())) {
    auto& t = tensor(i);
    TORCH_CHECK(t.dim() > 0, "zero-dimensional tensor (at position ", i,
                ") cannot be concatenated");
    if (!cat_should_skip_tensor(t)) {
      valid = i;
      dim = c10::maybe_wrap_dim(dim, t.dim());
      break;
    }
  }

  shape_ = {0};

  // If we found a valid tensor, check whether the input tensors
  // are compatible
  if (valid < ntensors()) {
    TORCH_CHECK(dim <= tensor(valid).dim(), "torch.cat(): dimension ", dim,
                "out of range");

    // Compute the output tensor size.
    // It should have the same shape as any other valid tensor,
    // except in the dimension 'dim'.
    int64_t size_at_dim = 0;
    for (const auto i : c10::irange(ntensors())) {
      const at::Tensor& t = tensor(i);
      if (!cat_should_skip_tensor(t)) {
        check_cat_shape_except_dim(valid, i, dim);
        size_at_dim += t.size(dim);
      }
    }

    shape_ = tensor(valid).sizes();
    shape_[dim] = size_at_dim;
  }
}

at::Tensor CatOpInferrer::infer_out(const at::ITensorListRef& tensors,
                                    int64_t dim) {
  for (auto& t : tensors) {
    add_input(t);
  }
  TORCH_CHECK(!inputs_.empty(),
              "torch.cat(): expected a non-empty list of Tensors");

  compute_shape(dim);
  compute_dtype();
  compute_memory_format();
  return malloc_output();
}

}  // namespace dipu
