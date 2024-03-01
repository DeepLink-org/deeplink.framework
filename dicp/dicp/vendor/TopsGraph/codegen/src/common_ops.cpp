#include "common_ops.h"

builder::Op enflame::Gather(std::shared_ptr<builder::Builder> hlir_builder,
                            builder::Op input, builder::Op index,
                            const int64_t dim, builder::Type gather_type) {
  // input and index shape
  auto input_shape = input.GetType().GetShape();
  auto index_shape = index.GetType().GetShape();

  // main compute
  // if (index.GetType().GetPrimitiveType().GetUnitBytes() == 8 &&
  //     index.GetType().GetShape().at(dim) <
  //     std::numeric_limits<uint32_t>::max()) {
  //   index = builder::Convert(index, builder::Type(index.GetType().GetShape(),
  //   builder::PrimitiveType::U32()));
  // }
  std::vector<int64_t> input_broadcast_dims;
  std::vector<int64_t> index_broadcast_dims;
  std::vector<int64_t> sizes;

  sizes.reserve(index_shape.size() + 1);
  for (size_t i = 0; i < index_shape.size(); ++i) {
    if (i < dim) {
      input_broadcast_dims.push_back(i);
      index_broadcast_dims.push_back(i);
    } else if (i == dim) {
      sizes.push_back(input_shape[i]);
      input_broadcast_dims.push_back(i);
      index_broadcast_dims.push_back(i + 1);
    } else {
      input_broadcast_dims.push_back(i + 1);
      index_broadcast_dims.push_back(i + 1);
    }
    sizes.push_back(index_shape[i]);
  }

  builder::Type mask_type(sizes, index.GetType().GetPrimitiveType());
  builder::Op mask = builder::Equal(
      builder::BroadcastInDim(index, index_broadcast_dims, mask_type),
      builder::Iota(hlir_builder, dim, mask_type));

  builder::Type masked_input_type(sizes, input.GetType().GetPrimitiveType());
  builder::Op masked_input = builder::Select(
      mask,
      builder::BroadcastInDim(input, input_broadcast_dims, masked_input_type),
      builder::ZerosLike(mask, input.GetType().GetPrimitiveType()));

  // add func binary_add
  hlir_builder->AddFunc("binary_add");
  builder::Type scalar_type(input.GetType().GetPrimitiveType());
  auto arg0 = hlir_builder->CreateInput(scalar_type, "binary_add");
  auto arg1 = hlir_builder->CreateInput(scalar_type, "binary_add");
  auto added = builder::Add(arg0, arg1);
  hlir_builder->SetOutput({added}, "binary_add");

  builder::Op scalar_zero =
      builder::Const(hlir_builder, 0.,
                     builder::Type(masked_input.GetType().GetPrimitiveType()));
  builder::Op res = builder::Reduce({masked_input}, {scalar_zero}, {dim},
                                    {"binary_add"}, "", gather_type);

  return res;
}

builder::Op enflame::BatchNorm(std::shared_ptr<builder::Builder> hlir_builder,
                               builder::Op& input, builder::Op& weight,
                               builder::Op& bias, builder::Op& running_mean,
                               builder::Op& running_var, int64_t channel_dim,
                               bool training, double momentum, double eps) {
  float eps_float = static_cast<float>(eps);
  float momentum_float = static_cast<float>(momentum);
  builder::Op batch_norm_op_res;
  std::vector<builder::Op> outputs;

  if (training) {
    std::vector<int64_t> input_shape = input.GetType().GetShape();
    std::vector<std::vector<int64_t>> res_shape = {
        input_shape, {input_shape[1]}, {input_shape[1]}};
    builder::PrimitiveType primitive_type = input.GetType().GetPrimitiveType();
    std::vector<builder::PrimitiveType> res_primitive_type = {
        primitive_type, primitive_type, primitive_type};
    builder::Type res_type(res_shape, res_primitive_type);
    batch_norm_op_res = builder::BatchNormTraining(
        input, weight, bias, eps_float, channel_dim, res_type);
  } else {
    batch_norm_op_res = builder::BatchNormInference(
        input, weight, bias, running_mean, running_var, eps_float, channel_dim);
  }
  auto output = builder::GetTupleElement(batch_norm_op_res, 0);
  auto save_mean = builder::GetTupleElement(batch_norm_op_res, 1);
  auto save_var = builder::GetTupleElement(batch_norm_op_res, 2);
  auto save_invstd = builder::Rsqrt(
      builder::Add(save_var, builder::FullLike(save_var, eps_float)));
  if (training) {
    // auto res = builder::Tuple({output, save_mean, save_var,
    // running_mean_updated, running_var_updated}); output
    // res.SetAttribute("op_name", builder::Attribute("TorchBatchNorm"));
    builder::Op running_mean_updated = builder::Add(
        builder::Mul(save_mean, builder::FullLike(save_mean, momentum_float)),
        builder::Mul(running_mean,
                     builder::FullLike(running_mean, 1.0f - momentum_float)));
    auto input_shape = input.GetType().GetShape();
    decltype(input_shape) input_without_channel_shape(input_shape.begin(),
                                                      input_shape.end());
    input_without_channel_shape.erase(input_without_channel_shape.begin() +
                                      static_cast<size_t>(channel_dim));
    int64_t input_without_channel_number = std::accumulate(
        input_without_channel_shape.begin(), input_without_channel_shape.end(),
        1, std::multiplies<int64_t>());
    // std::cout << "save_var_reduce_channel_number: " <<
    // input_without_channel_number << std::endl;
    builder::Op unbiased_var = builder::Div(
        builder::Mul(save_var,
                     builder::FullLike(save_var, input_without_channel_number,
                                       save_var.GetType().GetPrimitiveType())),
        builder::FullLike(
            save_var, input_without_channel_number - static_cast<int64_t>(1),
            save_var.GetType().GetPrimitiveType()));

    builder::Op running_var_updated = builder::Add(
        builder::Mul(unbiased_var,
                     builder::FullLike(unbiased_var, momentum_float)),
        builder::Mul(running_var,
                     builder::FullLike(running_var, 1.0f - momentum_float)));
    outputs.push_back(output);
    outputs.push_back(save_mean);
    outputs.push_back(save_invstd);
    outputs.push_back(running_mean_updated);
    outputs.push_back(running_var_updated);
    // std::vector<builder::Op> outputs{output, save_mean, save_invstd,
    // running_mean_updated, running_var_updated};
  } else {
    outputs.push_back(output);
    outputs.push_back(save_mean);
    outputs.push_back(save_invstd);
    outputs.push_back(running_mean);
    outputs.push_back(running_var);
    // std::vector<builder::Op> outputs{output, save_mean, save_invstd,
    // running_mean,running_var};
  }

  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  for (uint i = 0; i < outputs.size(); i++) {
    tuple_shape.push_back(outputs[i].GetType().GetShape());
    tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
  }

  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto result = builder::Tuple(outputs, outputs_type);

  return result;
}

builder::Op enflame::GroupNorm(std::shared_ptr<builder::Builder> hlir_builder,
                               builder::Op input, builder::Op scale,
                               builder::Op bias, int64_t num_groups,
                               float epsilon, bool is_clast) {
  std::vector<int64_t> spatial_dimensions;
  builder::DimensionsLayout layout;
  if (is_clast) {
    spatial_dimensions = {1, 2};
    layout = builder::DimensionsLayout(0, 3, spatial_dimensions);
  } else {
    spatial_dimensions = {2, 3};
    layout = builder::DimensionsLayout(0, 1, spatial_dimensions);
  }
  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  tuple_dtype.push_back(input.GetType().GetPrimitiveType());
  tuple_dtype.push_back(scale.GetType().GetPrimitiveType());
  tuple_dtype.push_back(bias.GetType().GetPrimitiveType());
  std::vector<int64_t> input_shape = input.GetType().GetShape();
  std::vector<int64_t> shape{input_shape[0], num_groups};
  tuple_shape.push_back(input_shape);
  tuple_shape.push_back(shape);
  tuple_shape.push_back(shape);
  builder::Type group_norm_type(tuple_shape, tuple_dtype);

  builder::Op group_norm = builder::GroupNorm(input, scale, bias, num_groups,
                                              epsilon, layout, group_norm_type);
  return group_norm;
}

builder::Op enflame::LayerNorm(std::shared_ptr<builder::Builder> hlir_builder,
                               builder::Op input, builder::Op scale,
                               builder::Op bias, float epsilon, bool is_clast) {
  int64_t axis = 2;
  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  tuple_dtype.push_back(input.GetType().GetPrimitiveType());
  tuple_dtype.push_back(scale.GetType().GetPrimitiveType());
  tuple_dtype.push_back(bias.GetType().GetPrimitiveType());
  auto input_shape = input.GetType().GetShape();
  std::vector<int64_t> shape{input_shape[0], input_shape[1]};
  tuple_shape.push_back(input_shape);
  tuple_shape.push_back(shape);
  tuple_shape.push_back(shape);
  builder::Type layer_norm_type(tuple_shape, tuple_dtype);

  builder::Op layer_norm =
      builder::LayerNorm(input, scale, bias, axis, epsilon, layer_norm_type);
  return layer_norm;
}

builder::Op enflame::UpsampleNearest2d(
    std::shared_ptr<builder::Builder> hlir_builder, builder::Op input,
    std::vector<int64_t> output_size, float scales_h, float scales_w,
    bool is_clast) {
  auto input_shape = input.GetType().GetShape();
  std::vector<int64_t> roi_values(input_shape.size(), 0);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    roi_values.push_back(input_shape[i] - 1);
  }
  builder::Op roi_op = builder::Const(
      hlir_builder, static_cast<void*>(roi_values.data()),
      builder::Type({roi_values.size()}, builder::PrimitiveType::S64()));
  builder::Op scales_op;
  builder::Op sizes_op;
  if (scales_h && scales_w) {
    // input op layout is HNWC, convert scales form [scales_h, scales_w] to
    // [1, scales_h, scales_w, 1]
    std::vector<float> temp(4, 1.0);
    if (is_clast) {
      temp[1] = scales_h, temp[2] = scales_w;
    } else {
      temp[2] = scales_h, temp[3] = scales_w;
    }
    scales_op =
        builder::Const(hlir_builder, static_cast<void*>(temp.data()),
                       builder::Type({4}, builder::PrimitiveType::F32()));
    sizes_op = sizes_op =
        builder::Const(hlir_builder, nullptr,
                       builder::Type({0}, builder::PrimitiveType::S64()));
  } else {
    scales_op =
        builder::Const(hlir_builder, nullptr,
                       builder::Type({0}, builder::PrimitiveType::F32()));
    sizes_op = builder::Const(
        hlir_builder, static_cast<void*>(input_shape.data()),
        builder::Type({input_shape.size()}, builder::PrimitiveType::S64()));
  }
  builder::Op resize = builder::Resize(input, roi_op, scales_op, sizes_op, 0, 1,
                                       false, 3, 0.0, -0.75);
  return resize;
}

builder::Op enflame::Convolution(std::shared_ptr<builder::Builder> hlir_builder,
                                 std::vector<builder::Op> inputs, int64_t group,
                                 std::vector<int64_t> stride,
                                 std::vector<int64_t> padding,
                                 std::vector<int64_t> dilation, bool is_clast) {
  std::string auto_pad = "NOTSET";
  std::string layout;
  if (is_clast) {
    layout = "NHWC";
  } else {
    layout = "NCHW";
  }
  builder::Op result = builder::Conv2D(inputs, group, auto_pad, layout, stride,
                                       padding, dilation);
  result.SetAttribute("op_type", builder::Attribute("Conv2DInference"));
  return result;
}

builder::Op enflame::ViewAsComplex(
    std::shared_ptr<builder::Builder> hlir_builder, builder::Op input,
    const std::vector<int64_t> shape) {
  auto out_shape = shape;
  out_shape.push_back(2);
  int out_shape_size = out_shape.size();

  std::vector<int64_t> view_as_complex_2_part0_start_indices(out_shape_size, 0);
  auto view_as_complex_2_part0_limit_indices = out_shape;
  view_as_complex_2_part0_limit_indices[out_shape_size - 1]--;
  std::vector<int64_t> view_as_complex_2_part1_start_indices(out_shape_size, 0);
  view_as_complex_2_part1_start_indices[out_shape_size - 1] = 1;
  std::vector<int64_t> view_as_complex_2_stride(out_shape_size, 1);

  builder::Op view_as_complex_2_split0 = builder::Slice(
      input, view_as_complex_2_part0_start_indices,
      view_as_complex_2_part0_limit_indices, view_as_complex_2_stride);
  builder::Op view_as_complex_2_split1 =
      builder::Slice(input, view_as_complex_2_part1_start_indices, out_shape,
                     view_as_complex_2_stride);
  builder::Type view_as_complex_2_reshape_type(shape,
                                               builder::PrimitiveType::F32());
  builder::Op view_as_complex_2_tmp0 = builder::Reshape(
      view_as_complex_2_split0, view_as_complex_2_reshape_type);
  builder::Op view_as_complex_2_tmp1 = builder::Reshape(
      view_as_complex_2_split1, view_as_complex_2_reshape_type);
  std::vector<builder::Op> view_as_complex_2_outputs{view_as_complex_2_tmp0,
                                                     view_as_complex_2_tmp1};
  builder::Op view_as_complex_2 = builder::Tuple(view_as_complex_2_outputs);
  return view_as_complex_2;
}

builder::Op enflame::ViewAsReal(std::shared_ptr<builder::Builder> hlir_builder,
                                builder::Op input,
                                const std::vector<int64_t> shape) {
  std::vector<int64_t> real_shape = shape;
  real_shape.pop_back();
  real_shape.push_back(1);
  int cat_dim = real_shape.size() - 1;

  builder::Op view_as_real_5_real = builder::GetTupleElement(input, 0);
  builder::Op view_as_real_5_imag = builder::GetTupleElement(input, 1);
  builder::Type view_as_real_5_reshape_type(real_shape,
                                            builder::PrimitiveType::F32());
  builder::Op view_as_real_5_tmp0 =
      builder::Reshape(view_as_real_5_real, view_as_real_5_reshape_type);
  builder::Op view_as_real_5_tmp1 =
      builder::Reshape(view_as_real_5_imag, view_as_real_5_reshape_type);
  std::vector<builder::Op> view_as_real_5_real_imag = {view_as_real_5_tmp0,
                                                       view_as_real_5_tmp1};
  builder::Op view_as_real_5 =
      builder::Concatenate(view_as_real_5_real_imag, cat_dim);

  return view_as_real_5;
}

builder::Op enflame::ComplexMul(std::shared_ptr<builder::Builder> hlir_builder,
                                builder::Op lhs, builder::Op rhs) {
  builder::Op xreal = builder::GetTupleElement(lhs, 0);
  builder::Op ximag = builder::GetTupleElement(lhs, 1);
  builder::Op yreal = builder::GetTupleElement(rhs, 0);
  builder::Op yimag = builder::GetTupleElement(rhs, 1);
  builder::Op xreal_yreal = builder::Mul(xreal, yreal);
  builder::Op ximag_yimag = builder::Mul(ximag, yimag);
  builder::Op xreal_yimag = builder::Mul(xreal, yimag);
  builder::Op ximag_yreal = builder::Mul(ximag, yreal);
  builder::Op mul_real = builder::Sub(xreal_yreal, ximag_yimag);
  builder::Op mul_imag = builder::Add(xreal_yimag, ximag_yreal);
  std::vector<builder::Op> complex_mul_outputs{mul_real, mul_imag};
  builder::Op complex_mul = builder::Tuple(complex_mul_outputs);

  return complex_mul;
}
