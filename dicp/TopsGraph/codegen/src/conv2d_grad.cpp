#include "conv2d_grad.h"

const char *const kConv2D = "conv2d";
const char *const kConv2DGrad = "conv2d_grad";

static std::vector<int64_t> get_same_padding_value(int64_t dim,
                                                   int64_t ksize,
                                                   int64_t stride) {
  int64_t pad_along_dim = 0;
  if (dim % stride == 0) {
    pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
  } else {
    pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
  }
  int64_t pad_low = pad_along_dim / 2;
  int64_t pad_high = pad_along_dim - pad_low;
  std::vector<int64_t> padding{pad_low, pad_high};
  return std::move(padding);
}

static std::pair<int64_t, int64_t> get_backprop_filter_padding(
    int64_t input_dim,
    int64_t output_dim,
    int64_t kernel_size,
    int64_t stride,
    int64_t dilation,
    int64_t padding_before) {
  std::pair<int64_t, int64_t> padding_dim;
  int64_t expanded_output_size = (output_dim - 1) * stride + 1;
  int64_t padded_in_size = (kernel_size - 1) * dilation;
  padded_in_size += expanded_output_size;
  int64_t pad_total = padded_in_size - input_dim;
  // int64_t pad_before = padding_before;
  int64_t pad_before = std::max<int64_t>(pad_total / 2, 0);
  

  padding_dim = {pad_before, pad_total - pad_before};
  return padding_dim;
}

static std::pair<int64_t, int64_t> get_backprop_input_padding(int64_t input_dim,
                                                       int64_t output_dim,
                                                       int64_t kernel_size,
                                                       int64_t stride,
                                                       int64_t dilation,
                                                       int64_t padding_before) {
  std::pair<int64_t, int64_t> padding_dim;
  int64_t effective_filter_size = (kernel_size - 1) * dilation + 1;
  int64_t expanded_output_size = (output_dim - 1) * stride + 1;
  int64_t padded_out_size = input_dim + effective_filter_size - 1;
  int64_t pad_before = effective_filter_size - 1 - padding_before;
  int64_t pad_after = padded_out_size - expanded_output_size - pad_before;
  padding_dim = {pad_before, pad_after};
  return padding_dim;
}

builder::Op enflame::Conv2d_Grad(
    std::shared_ptr<builder::Builder> tmp_builder,
    builder::Op out_grad_,
    builder::Op input_,
    builder::Op filter_,
    std::vector<int64_t> bias_shape,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation) {
  // do not take bias in account because bias_grad will be calculated in
  // elementwise_add_grad input keys: Output@GRAD, Filter, Input output keytrs:
  // Filter@GRAD, Input@GRAD
  builder::Op out_grad = builder::Transpose(out_grad_, {0, 2, 3, 1});
  builder::Op input = builder::Transpose(input_, {0, 2, 3, 1});
  builder::Op filter = builder::Transpose(filter_, {2, 3, 1, 0});

  auto input_shape = input.GetType().GetShape();
  auto kernel_shape = filter.GetType().GetShape();
  auto output_shape = out_grad.GetType().GetShape();

  int64_t ih = input_shape[1];
  int64_t iw = input_shape[2];
  int64_t kh = kernel_shape[0];
  int64_t kw = kernel_shape[1];
  int64_t oh = output_shape[1];
  int64_t ow = output_shape[2];

  auto pad_h = get_same_padding_value(ih, kh, stride[0]);
  auto pad_w = get_same_padding_value(iw, kw, stride[1]);
  padding = {pad_h[0], pad_h[1], pad_w[0], pad_w[1]};

  // calculate filter_grad
  builder::Op filter_grad;
  if (true) {
    std::vector<int64_t> window_strides = dilation;
    for(uint i = 0; i < window_strides.size(); i++){
      if(window_strides[i] < 1){
        window_strides[i] = 1;
      }
    }
    std::vector<int64_t> rhs_dilation = stride;

    auto pad_h = get_backprop_filter_padding(
        ih, oh, kh, stride[0], dilation[0], padding[0]);
    auto pad_w = get_backprop_filter_padding(
        iw, ow, kw, stride[1], dilation[1], padding[2]);
    std::vector<std::vector<int64_t>> paddings = {{pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};
    builder::ConvDimensionNumbers dims_attr(
        /*input_batch_dimension=*/3,
        /*input_feature_dimension=*/0,
        /*input_spatial_dimensions=*/{1, 2},
        /*kernel_input_feature_dimension=*/0,
        /*kernel_output_feature_dimension=*/3,
        /*kernel_spatial_dimensions=*/{1, 2},
        /*output_batch_dimension=*/2,
        /*output_feature_dimension=*/3,
        /*output_spatial_dimensions=*/{0, 1});
        
    filter_grad = builder::Conv(input,
                                out_grad,
                                dims_attr,
                                /*window_strides=*/window_strides,
                                /*padding=*/paddings,
                                /*lhs_dilation=*/{1, 1},
                                /*rhs_dilation=*/rhs_dilation,
                                /*window_reversal=*/{},
                                /*auto_pad=*/"",
                                /*feature_group_count=*/1,
                                /*batch_group_count=*/1,
                                /*precision_config=*/{});
    filter_grad.SetAttribute("op_type",
                             builder::Attribute("Conv2DBackpropFilter"));
    // std::cout << filter_grad << std::endl;
    // std::cout << "---- debug filter_grad end" << std::endl;
    
    filter_grad = builder::Transpose(filter_grad, {3, 2, 0, 1});
  }

  // calculate input_grad
  builder::Op input_grad;
  if (true) {
    auto filter_reverse = builder::Reverse(filter, {0, 1}, filter.GetType());
    std::vector<int64_t> lhs_dilation = stride;
    std::vector<int64_t> rhs_dilation = dilation;

    auto pad_h = get_backprop_input_padding(
        ih, oh, kh, stride[0], dilation[0], padding[0]);
    auto pad_w = get_backprop_input_padding(
        iw, ow, kw, stride[1], dilation[1], padding[2]);
    
    std::vector<std::vector<int64_t>> paddings = {{pad_h.first, pad_h.second},
                                                  {pad_w.first, pad_w.second}};

    builder::ConvDimensionNumbers dims_attr(
        /*input_batch_dimension=*/0,
        /*input_feature_dimension=*/3,
        /*input_spatial_dimensions=*/{1, 2},
        /*kernel_input_feature_dimension=*/3,
        /*kernel_output_feature_dimension=*/2,
        /*kernel_spatial_dimensions=*/{0, 1},
        /*output_batch_dimension=*/0,
        /*output_feature_dimension=*/3,
        /*output_spatial_dimensions=*/{1, 2});

    input_grad = builder::Conv(out_grad,
                               filter_reverse,
                               dims_attr,
                               /*window_strides=*/{1, 1},
                               /*padding=*/paddings,
                               /*lhs_dilation=*/lhs_dilation,
                               /*rhs_dilation=*/rhs_dilation,
                               /*window_reversal=*/{},
                               /*auto_pad=*/"",
                               /*feature_group_count=*/1,
                               /*batch_group_count=*/1,
                               /*precision_config=*/{});
    input_grad.SetAttribute("op_type",
                            builder::Attribute("Conv2DBackpropInput"));
    // std::cout << input_grad << std::endl;
    // std::cout << "---- debug input_grad end" << std::endl;
    input_grad = builder::Transpose(input_grad, {0, 3, 1, 2});
  }

  // std::vector<int64_t> bias_grad_shape{}
  builder::Type bias_type(bias_shape, builder::PrimitiveType::F32());
  int bias_size = 1;
  for (uint i = 0; i < bias_shape.size(); i++) {
    bias_size = bias_size * bias_shape[i];
  }
  std::vector<int64_t> bias_data(bias_size, 0.0);
  auto bias_grad = builder::Const(tmp_builder, static_cast<void *>(bias_data.data()), bias_type);

  std::vector<builder::Op> outputs{input_grad, filter_grad, bias_grad};

  // return outputs;
  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  for (uint i = 0; i < outputs.size(); i++) {
    tuple_shape.push_back(outputs[i].GetType().GetShape());
    tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
  }

  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto result = builder::Tuple(outputs);
  
  return result;
}