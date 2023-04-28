#pragma once

#include <memory>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>

#include "dtu_utils.h"

namespace enflame {
builder::Op Gather(
    std::shared_ptr<builder::Builder> tmp_builder,
    builder::Op input,
    builder::Op index,
    const int64_t dim,
    builder::Type gather_type);

static void PadToSize(builder::Op& operand, const std::vector<int64_t>& target_shape, builder::Op& output, builder::Op& pad_value) {
  bool has_padding = false;
  auto operand_shape = operand.GetType().GetShape();
  std::vector<int64_t> edge_padding_low;
  std::vector<int64_t> edge_padding_high;
  std::vector<int64_t> interior_padding;
  for (size_t i = 0; i < target_shape.size(); i++) {
    edge_padding_low.push_back(0);
    interior_padding.push_back(0);
    auto diff_in_high = target_shape[i] - operand_shape.at(i);
    edge_padding_high.push_back(diff_in_high);
    has_padding = has_padding || diff_in_high != 0;
  }
  if (has_padding) {
    output = builder::Pad(operand, pad_value, 0, edge_padding_low, edge_padding_high, interior_padding);
  } else {
    std::cout << "No need padding to size, weird!" << std::endl;
    output = operand;
  }
}

template<typename T>
builder::Op Scatter(
    std::shared_ptr<builder::Builder> hlir_builder,
    builder::Op& self,
    const int64_t dim,
    builder::Op& index,
    const T scalar_value) {
  
  builder::PrimitiveType src_dtype = self.GetType().GetPrimitiveType();
  builder::Op src = builder::FullLike(index, scalar_value, src_dtype);
  
  auto neg_inf = std::numeric_limits<T>::lowest();
  auto self_shape = self.GetType().GetShape();
  auto index_shape = index.GetType().GetShape();
  std::vector<int64_t> index_broadcast_dims;
  std::vector<int64_t> sizes;
  const auto rank = index_shape.size();
  sizes.reserve(rank + 1);
  for (int64_t i = 0; i < index_shape.size(); ++i) {
    if (i < dim) {
      index_broadcast_dims.push_back(i);
    } else {
      if (i == dim) {
        sizes.push_back(self_shape.at(i));
      }
      index_broadcast_dims.push_back(i + 1);
    }
    sizes.push_back(index_shape.at(i));
  }

  builder::Type mask_type(sizes, index.GetType().GetPrimitiveType());
  builder::Op mask = builder::Equal(
    builder::BroadcastInDim(index, index_broadcast_dims, mask_type),
    builder::Iota(hlir_builder, dim, mask_type)
  );
  builder::Type selected_src_type(sizes, src.GetType().GetPrimitiveType());
  builder::Op selected_src = builder::Select(
    mask,
    builder::BroadcastInDim(src, index_broadcast_dims, selected_src_type),
    builder::FullLike(mask, neg_inf, self.GetType().GetPrimitiveType())
  );

  // add func binary_max
  hlir_builder->AddFunc("binary_max");
  builder::Type scalar_type(self.GetType().GetPrimitiveType());
  auto max_lhs = hlir_builder->CreateInput(scalar_type, "binary_max");
  auto max_rhs = hlir_builder->CreateInput(scalar_type, "binary_max");
  auto max_res = builder::Max(max_lhs, max_rhs);
  hlir_builder->SetOutput({max_res}, "binary_max");

  builder::Op scalar_neg_inf = builder::Const(hlir_builder, neg_inf, builder::Type(self.GetType().GetPrimitiveType()));
  builder::Op reduced_selected_src = builder::Reduce(
    {selected_src},
    {scalar_neg_inf},
    {dim + 1},
    {"binary_max"}
  );

  // add func binary_or
  hlir_builder->AddFunc("binary_or");
  builder::Type bool_scalar_type(builder::PrimitiveType::PRED());
  auto binary_or_arg0 = hlir_builder->CreateInput(bool_scalar_type, "binary_or");
  auto binary_or_arg1 = hlir_builder->CreateInput(bool_scalar_type, "binary_or");
  auto binary_or_result = builder::Or(binary_or_arg0, binary_or_arg1);
  hlir_builder->SetOutput({binary_or_result}, "binary_or");

  builder::Op scalar_false = builder::Const(hlir_builder, false, builder::Type(builder::PrimitiveType::PRED()));
  builder::Op reduced_mask = builder::Reduce(
    {mask},
    {scalar_false},
    {dim + 1},
    {"binary_or"}
  );

  // check whether scatter result requires padding
  bool requires_padding = false;
  for (size_t i = 0; i < self_shape.size(); ++i) {
    if (self_shape.at(i) > index_shape.at(i)) {
      requires_padding = true;
      break;
    } else if (i != dim) {
      if (self_shape.at(i) != index_shape.at(i)) {
        requires_padding = true;
        break;
      }
    }
  }
  if (requires_padding) {
    PadToSize(reduced_selected_src, self_shape, reduced_selected_src, scalar_neg_inf);
    PadToSize(reduced_mask, self_shape, reduced_mask, scalar_false);
  }

  builder::Op res = builder::Select(reduced_mask, reduced_selected_src, self);
  return res;
}

builder::Op BatchNorm(
    std::shared_ptr<builder::Builder> hlir_builder,
    builder::Op& input,
    builder::Op& weight,
    builder::Op& bias,
    builder::Op& running_mean,
    builder::Op& running_var,
    int64_t channel_dim,
    bool training,
    double momentum,
    double eps);
}  // namespace enflame