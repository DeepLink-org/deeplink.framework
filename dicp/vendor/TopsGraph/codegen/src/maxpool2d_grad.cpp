#include "maxpool2d_grad.h"

builder::Op enflame::MaxPool2D(
    std::shared_ptr<builder::Builder> tmp_builder,
    builder::Op input,
    std::vector<long> ksize,
    std::vector<long> strides,
    std::vector<long> padding,
    std::vector<long> shape) {
    auto res1 = builder::MaxPool2D(input, ksize, false, false, "NOTSET", "NCHW", strides, padding, {});
    
    builder::Type pool_type(shape, builder::PrimitiveType::S64());
    
    int size = 1;

    for (uint i = 0; i < shape.size(); i++) {
        size = size * shape[i];
    }
    std::vector<int64_t> pool_data(size, 1.0);
    auto res2 = builder::Const(tmp_builder, static_cast<void *>(pool_data.data()), pool_type);

    std::vector<builder::Op> outputs{res1, res2};

    auto res = builder::Tuple(outputs);

    return res;
}

builder::Op enflame::MaxPool2D_Grad(
    std::shared_ptr<builder::Builder> tmp_builder,
    builder::Op out_grad,
    builder::Op in,
    std::vector<int> ksize,
    std::vector<int> strides,
    std::vector<int> padding) {
    auto input_shape = in.GetType().GetShape();
    auto ptype = in.GetType().GetPrimitiveType();
    
    int64_t kh = static_cast<int64_t>(ksize[0]);
    int64_t kw = static_cast<int64_t>(ksize[1]);

    if (padding.size() == 1) {
        padding = {padding[0], padding[0], padding[0], padding[0]};
    } else if (padding.size() == 2) {
        padding = {padding[0], padding[0], padding[1], padding[1]};
    } else if (padding.size() == 8) {
        if (true) {
            padding = {padding[4], padding[5], padding[6], padding[7]};
        }
    }

    std::vector<int64_t> kernel_shape = {kh, kw};
    bool do_transpose = in.GetType().GetShape().size() == 4;

    // do_transpose = true;
    if (true) {
        std::vector<int64_t> window_dimensions;
        std::vector<int64_t> window_strides;
        std::vector<std::vector<int64_t>> paddings;

        if (true) {
            if (do_transpose) {
              window_dimensions = {1, kh, kw, 1};
              window_strides = {1, strides[0], strides[1], 1};
              paddings = {{0, 0}, {padding[0], padding[1]},
                          {padding[2], padding[3]}, {0, 0}};
            } else {
              window_dimensions = {1, 1, kh, kw};
              window_strides = {1, 1, strides[0], strides[1]};
              paddings = {{0, 0}, {0, 0},
                          {padding[0], padding[1]}, {padding[2], padding[3]}};
            }
        } 

        std::vector<float> zero_data{0.0};
        void* data = static_cast<void*>(zero_data.data());
        builder::Type scalar_type(ptype);
        auto zero = builder::Const(tmp_builder, data, scalar_type);

        tmp_builder->AddFunc("select");
        auto pred_type = builder::PrimitiveType::PRED();
        auto arg0 = tmp_builder->CreateInput(scalar_type, "select");
        auto arg1 = tmp_builder->CreateInput(scalar_type, "select");
        auto ge = builder::Compare(arg0, arg1, "GE", {}, {{}, pred_type});
        tmp_builder->SetOutput({ge}, "select");

        tmp_builder->AddFunc("scatter");
        arg0 = tmp_builder->CreateInput(scalar_type, "scatter");
        arg1 = tmp_builder->CreateInput(scalar_type, "scatter");
        auto sum = arg0 + arg1;
        tmp_builder->SetOutput({sum}, "scatter");

        if (do_transpose) {
            in = builder::Transpose(in, {0, 2, 3, 1});
            out_grad = builder::Transpose(out_grad, {0, 2, 3, 1});

            auto tmp_op = builder::SelectAndScatter(in, out_grad, zero,
                                                    {"select", "scatter"},
                                                    window_dimensions,
                                                    window_strides,
                                                    paddings,
                                                    in.GetType());

            return builder::Transpose(tmp_op, {0, 3, 1, 2});
        } else {
            return builder::SelectAndScatter(in, out_grad, zero,
                                            {"select", "scatter"},
                                            window_dimensions,
                                            window_strides,
                                            paddings,
                                            in.GetType());
        }
    }
}