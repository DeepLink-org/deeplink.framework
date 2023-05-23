// Copyright (c) 2023, DeepLink.
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_ADAPTOR_HPP_
#define DIOPI_ADAPTOR_HPP_
#include <iostream>
#include <vector>
#include <diopi/diopirt.h>
#include <diopi/functions.h>

namespace diopiadaptor{

inline std::vector<int64_t> calcStrides(int ndims, diopiSize_t size, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    std::vector<int64_t> strides;
    strides.resize(ndims);
    int64_t st = 1;
    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = ndims; i > 0; --i) {
            strides[i - 1] = st;
            if (size.data[i - 1] == 0) continue;
            if (size.data[i - 1] == -1) st = -1;
            if (st != -1) st *= size.data[i - 1];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        for (auto k : {1, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) st *= size.data[k];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        for (auto k : {1, 4, 3, 2, 0}) {
            strides[k] = st;
            if (size.data[k] == 0) {
                continue;
            }
            if (size.data[k] == -1) st = -1;
            if (st != -1) {
                st *= size.data[k];
            }
        }

    }
    else {
        // PARROTS_THROW(InvalidArgs) <<
        //         "Invalid MemoryFormat " << memoryFormatName(format);
    }
    return strides;
}

inline bool isLikeChannelsLast(diopiConstTensorHandle_t tensor, bool checkContiguous, diopiMemoryFormat_t format = diopiMemoryFormat_t::ChannelsLast) {
    diopiSize_t shape, stride;
    diopiGetTensorShape(tensor, &shape);
    diopiGetTensorStride(tensor, &stride);
    if (shape.len != 4) return false;
    int64_t totalSize = 1;
    for (int64_t i = 0; i < shape.len; ++i) {
        totalSize *= shape.data[i];
    }
    if (totalSize == 0) return false;
    if (stride.data[0] == stride.data[1]) return false;
    if (checkContiguous) {
        auto realStride = calcStrides(shape.len, shape, format);
        for (int i = 0; i < stride.len; ++i) {
            if (i >= realStride.size() || realStride[i] != stride.data[i]) {
                return false;
            }
        }
        return true;
    } else {
        int64_t st = 1;
        std::vector<int> orders;
        if (format == diopiMemoryFormat_t::ChannelsLast)
            orders = {1, 3, 2, 0};
        else if (format == diopiMemoryFormat_t::ChannelsLast3d)
            orders = {1, 4, 3, 2, 0};
        for (auto k : orders) {
            if (stride.data[k] < st) return false;
            st = stride.data[k] * shape.data[k];
        }
        return true;
    }
}

inline diopiMemoryFormat_t probableMemoryFormat(diopiConstTensorHandle_t tensor, bool exactMatch = false) {
    return isLikeChannelsLast(tensor, exactMatch) ? diopiMemoryFormat_t::ChannelsLast
        : (isLikeChannelsLast(tensor, exactMatch, diopiMemoryFormat_t::ChannelsLast3d) ? diopiMemoryFormat_t::ChannelsLast3d
        : diopiMemoryFormat_t::Contiguous);
}

inline bool isContiguous(diopiSize_t size, diopiSize_t stride, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous) {
    int64_t totalSize = 1;
    for (int64_t i = 0; i < size.len; ++i) {
        totalSize *= size.data[i];
    }
    if (totalSize == 0) return false;
    if (format == diopiMemoryFormat_t::ChannelsLast && size.len != 4) return false;
    auto st = calcStrides(size.len, size, format);
    for (int64_t i = 0; i < size.len; ++i) {
        if (st[i] != stride.data[i] && size.data[i] > 1) return false;
    }
    return true;
}

static std::vector<diopiMemoryFormat_t> defaultFormats{diopiMemoryFormat_t::Contiguous, diopiMemoryFormat_t::ChannelsLast};

class NoCast {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {

            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

class Default {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            case diopiDtype_t::diopi_dtype_int64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_int32;
                 break;
            case diopiDtype_t::diopi_dtype_float64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_float32;
                 break;
            case diopiDtype_t::diopi_dtype_bool:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_int32;
                 break;
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

class CastFloatOnly {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            case diopiDtype_t::diopi_dtype_int64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_int32;
                 break;
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

class LogicOp {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            case diopiDtype_t::diopi_dtype_int64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_int32;
                 break;
            case diopiDtype_t::diopi_dtype_float64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_int32;
                 break;
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

class CommonCast {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            case diopiDtype_t::diopi_dtype_int64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_int32;
                 break;
            case diopiDtype_t::diopi_dtype_float64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_float32;
                 break;
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

class diopiAddInputCast {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            case diopiDtype_t::diopi_dtype_float64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_float32;
                 break;
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

class diopiAddOtherCast {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            case diopiDtype_t::diopi_dtype_float64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_float32;
                 break;
            case diopiDtype_t::diopi_dtype_int64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_float32;
                 break;
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

class diopiDropoutCast {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            case diopiDtype_t::diopi_dtype_int64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_int32;
                 break;
            case diopiDtype_t::diopi_dtype_float64:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_float32;
                 break;
            case diopiDtype_t::diopi_dtype_bool:
                 convert = true;
                 targetDtype = diopiDtype_t::diopi_dtype_uint8;
                 break;
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};

template<class T, class strategy = NoCast, bool needContiguous = false>
inline int castImpl(diopiContextHandle_t ctx, T src, T* dst,
                    std::vector<diopiMemoryFormat_t> supportMemoryFormat = defaultFormats) {
    if (src == nullptr || src == 0) {
        *dst = src;
        return 0;
    }
    diopiDtype_t srcDtype, dstDtype;
    diopiGetTensorDtype(src, &srcDtype);
    diopiSize_t size, stride, dstStride;
    diopiGetTensorShape(src, &size);
    diopiGetTensorStride(src, &stride);
    diopiDevice_t device;
    diopiGetTensorDevice(src, &device);

    bool convertDtype = strategy::getDstDtype(srcDtype, dstDtype);
    auto memoryFormat = probableMemoryFormat(src);
    bool convertFormat = true;
    for (int i = 0; i < supportMemoryFormat.size(); ++i) {
        if (supportMemoryFormat[i] == memoryFormat) {
            convertFormat = false;
            break;
        }
    }
    bool contiguous = needContiguous && isContiguous(size, stride, memoryFormat);
    int convertType = 0;
    if (!convertFormat) {
        dstStride = stride;
    } else {
        auto strides_v = calcStrides(size.len, size, memoryFormat);
        dstStride.len = strides_v.size();
        dstStride.data = strides_v.data();
    }
    if (convertDtype) {
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &size, &stride, dstDtype, device);
        diopiCastDtype(ctx, tmp, src);
        *dst = tmp;
        convertType = 1;
    } else {
        *dst = src;
    }
    convertType = convertType << 1;
    if (convertFormat || !contiguous) {
        diopiTensorHandle_t tmp = nullptr;
        diopiRequireTensor(ctx, &tmp, &size, &stride, dstDtype, device);
        diopiCopyInp(ctx, *dst, tmp);
        *dst = tmp;
        convertType = convertType + 1;
    }
    if (convertType == 0) {
        *dst = src;
    }
    return convertType;
}

template <typename Adaptor, typename... Args>
void dispatch_diopi(diopiContextHandle_t ctx, Args&&... args) {
    auto adaptor = Adaptor();
    adaptor(ctx, std::forward<Args>(args)...);
}

template<class strategy = NoCast, bool isContiguous = false>
class DiopiTensorWrapper {
private:
    diopiContextHandle_t ctx_;
    diopiTensorHandle_t payload_;
    diopiTensorHandle_t tmp_ = nullptr;
    int convertType_ = 0;

public:
    DiopiTensorWrapper(diopiContextHandle_t ctx, diopiTensorHandle_t payload,
                       std::vector<diopiMemoryFormat_t> supportMemoryFormat = defaultFormats)
                       : ctx_(ctx)
                       , payload_(payload) {
        convertType_ = castImpl<diopiTensorHandle_t, strategy, isContiguous>(ctx, payload_, &tmp_, supportMemoryFormat);
    }

    ~DiopiTensorWrapper() {
        if (convertType_ == 0) {
            if (tmp_) {
                payload_ = tmp_;
            }
            return;
        }
        if (convertType_ == 1){
            diopiCopyInp(ctx_, tmp_, payload_);
        } else if (convertType_ == 2) {
            diopiCastDtype(ctx_, payload_, tmp_);
        } else {
            diopiDtype_t dtype;
            diopiGetTensorDtype(tmp_, &dtype);
            diopiSize_t size, stride, dstStride;
            diopiGetTensorShape(payload_, &size);
            diopiGetTensorStride(payload_, &stride);
            diopiDevice_t device;
            diopiGetTensorDevice(payload_, &device);
            diopiTensorHandle_t tmp = nullptr;
            diopiRequireTensor(ctx_, &tmp, &size, &stride, dtype, device);
            diopiCopyInp(ctx_, tmp_, tmp);
            diopiCastDtype(ctx_, payload_, tmp);
        }
    }

public:
    operator diopiTensorHandle_t() {
        return tmp_;
    }
};

inline diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiConvolution2d(ctx, outWrapper, newInput, newWeight, newBias, stride, padding, dilation, groups);
}

inline diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiConvolution2dBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
}

inline diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_meanWrapper = DiopiTensorWrapper<CommonCast>(ctx, save_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_invstdWrapper = DiopiTensorWrapper<CommonCast>(ctx, save_invstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto running_meanWrapper = DiopiTensorWrapper<CommonCast>(ctx, running_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto running_varWrapper = DiopiTensorWrapper<CommonCast>(ctx, running_var, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBatchNorm(ctx, outWrapper, save_meanWrapper, save_invstdWrapper, newInput, newWeight, newBias, running_meanWrapper, running_varWrapper, training, momentum, eps);
}

inline diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight,newRunning_mean,newRunning_var,newSave_mean,newSave_invstd;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, running_mean, &newRunning_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, running_var, &newRunning_var, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, save_mean, &newSave_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, save_invstd, &newSave_invstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBatchNormBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, newRunning_mean, newRunning_var, newSave_mean, newSave_invstd, training, eps);
}

inline diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRelu(ctx, outWrapper, newInput);
}

inline diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiReluInp(ctx, inputWrapper);
}

inline diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardtanh(ctx, outWrapper, newInput, min_val, max_val);
}

inline diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardtanhInp(ctx, inputWrapper, min_val, max_val);
}

inline diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardtanhBackward(ctx, grad_inputWrapper, newGrad_output, newInput, min_val, max_val);
}

inline diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiHardswish(ctx, out, input);
}

inline diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiHardswishInp(ctx, input);
}

inline diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {



    return ::diopiHardswishBackward(ctx, grad_input, grad_output, input);
}

inline diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {



    return ::diopiThreshold(ctx, out, input, threshold, value);
}

inline diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {



    return ::diopiThresholdInp(ctx, input, threshold, value);
}

inline diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {



    return ::diopiThresholdBackward(ctx, grad_input, grad_output, input, threshold);
}

inline diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGelu(ctx, outWrapper, newInput, approximate);
}

inline diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const char* approximate) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeluBackward(ctx, grad_inputWrapper, newGrad_output, newInput, approximate);
}

inline diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeakyRelu(ctx, outWrapper, newInput, negative_slope);
}

inline diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeakyReluInp(ctx, inputWrapper, negative_slope);
}

inline diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeakyReluBackward(ctx, grad_inputWrapper, newGrad_output, newInput, negative_slope, input_is_result);
}

inline diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAvgPool2d(ctx, outWrapper, newInput, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

inline diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAvgPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

inline diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool2d(ctx, outWrapper, newInput, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<CommonCast>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool2dWithIndices(ctx, outWrapper, indicesWrapper, newInput, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    diopiConstTensorHandle_t newGrad_output,newInput,newIndices;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, kernel_size, stride, padding, dilation, ceil_mode, newIndices);
}

inline diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveAvgPool2d(ctx, outWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveAvgPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput);
}

inline diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool2d(ctx, outWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {



    return ::diopiAdaptiveMaxPool2dWithIndices(ctx, out, indices, input, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    diopiConstTensorHandle_t newGrad_output,newInput,newIndices;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newIndices);
}

inline diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,diopiDropoutCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::Contiguous});
    auto outWrapper = DiopiTensorWrapper<diopiDropoutCast, true>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::Contiguous});
    auto maskWrapper = DiopiTensorWrapper<diopiDropoutCast>(ctx, mask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::Contiguous});
    return ::diopiDropout(ctx, outWrapper, maskWrapper, newInput, p, train);
}

inline diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto maskWrapper = DiopiTensorWrapper<CommonCast>(ctx, mask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDropoutInp(ctx, inputWrapper, maskWrapper, p, train);
}

inline diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newInput,newTarget;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMSELoss(ctx, outWrapper, newInput, newTarget, reduction);
}

inline diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMSELossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, reduction);
}

inline diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs, diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction) {



    return ::diopiSigmoidFocalLoss(ctx, out, inputs, targets, alpha, gamma, reduction);
}

inline diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction) {



    return ::diopiSigmoidFocalLossBackward(ctx, grad_output, input, target, grad_input, gamma, alpha, reduction);
}

inline diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    diopiConstTensorHandle_t newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCrossEntropyLoss(ctx, outWrapper, newInput, newTarget, newWeight, reduction, ignore_index, label_smoothing);
}

inline diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCrossEntropyLossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, newWeight, reduction, ignore_index, label_smoothing);
}

inline diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    diopiConstTensorHandle_t newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNLLLoss(ctx, outWrapper, newInput, newTarget, newWeight, reduction, ignore_index);
}

inline diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNLLLossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, newWeight, reduction, ignore_index);
}

inline diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newInput,newTarget,newWeight,newPos_weight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, pos_weight, &newPos_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBCEWithLogits(ctx, outWrapper, newInput, newTarget, newWeight, newPos_weight, reduction);
}

inline diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget,newWeight,newPos_weight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, pos_weight, &newPos_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBCEWithLogitsBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, newWeight, newPos_weight, reduction);
}

inline diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {



    return ::diopiBCELoss(ctx, out, input, target, weight, reduction);
}

inline diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {



    return ::diopiBCELossBackward(ctx, grad_input, grad_output, input, target, weight, reduction);
}

inline diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiSign(ctx, out, input);
}

inline diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiAbsInp(ctx, input);
}

inline diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAbs(ctx, outWrapper, newInput);
}

inline diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNegInp(ctx, inputWrapper);
}

inline diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeg(ctx, outWrapper, newInput);
}

inline diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiFloorInp(ctx, inputWrapper);
}

inline diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiFloor(ctx, outWrapper, newInput);
}

inline diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSqrtInp(ctx, inputWrapper);
}

inline diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSqrt(ctx, outWrapper, newInput);
}

inline diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiRsqrtInp(ctx, input);
}

inline diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiRsqrt(ctx, out, input);
}

inline diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSinInp(ctx, inputWrapper);
}

inline diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSin(ctx, outWrapper, newInput);
}

inline diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCosInp(ctx, inputWrapper);
}

inline diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCos(ctx, outWrapper, newInput);
}

inline diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTanhInp(ctx, inputWrapper);
}

inline diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTanh(ctx, outWrapper, newInput);
}

inline diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTanhBackward(ctx, grad_inputWrapper, newGrad_output, newOutput);
}

inline diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoidInp(ctx, inputWrapper);
}

inline diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoid(ctx, outWrapper, newInput);
}

inline diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoidBackward(ctx, grad_inputWrapper, newGrad_output, newOutput);
}

inline diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiSiluInp(ctx, input);
}

inline diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiSilu(ctx, out, input);
}

inline diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {



    return ::diopiSiluBackward(ctx, grad_input, grad_output, input);
}

inline diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiExpInp(ctx, input);
}

inline diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiExp(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogInp(ctx, inputWrapper);
}

inline diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog2Inp(ctx, inputWrapper);
}

inline diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog2(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiLog10Inp(ctx, input);
}

inline diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiLog10(ctx, out, input);
}

inline diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiErfInp(ctx, input);
}

inline diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiErf(ctx, out, input);
}

inline diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {



    return ::diopiPowScalar(ctx, out, input, exponent);
}

inline diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPow(ctx, outWrapper, newInput, exponent);
}

inline diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPowInp(ctx, inputWrapper, exponent);
}

inline diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    diopiConstTensorHandle_t newInput,newExponent;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, exponent, &newExponent, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPowTensor(ctx, outWrapper, newInput, newExponent);
}

inline diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    diopiConstTensorHandle_t newExponent;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, exponent, &newExponent, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPowInpTensor(ctx, inputWrapper, newExponent);
}

inline diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,diopiAddInputCast>(ctx, input, &newInput);
    castImpl<diopiConstTensorHandle_t,diopiAddOtherCast>(ctx, other, &newOther);
    auto outWrapper = DiopiTensorWrapper<diopiAddOtherCast, true>(ctx, out);
    return ::diopiAdd(ctx, outWrapper, newInput, newOther, alpha);
}

inline diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddInp(ctx, inputWrapper, newOther, alpha);
}

inline diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddScalar(ctx, outWrapper, newInput, other, alpha);
}

inline diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddInpScalar(ctx, inputWrapper, other, alpha);
}

inline diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSub(ctx, outWrapper, newInput, newOther, alpha);
}

inline diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSubInp(ctx, inputWrapper, newOther, alpha);
}

inline diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSubScalar(ctx, outWrapper, newInput, other, alpha);
}

inline diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSubInpScalar(ctx, inputWrapper, other, alpha);
}

inline diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMul(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiMulInp(ctx, input, other);
}

inline diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {



    return ::diopiMulScalar(ctx, out, input, other);
}

inline diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {



    return ::diopiMulInpScalar(ctx, input, other);
}

inline diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDiv(ctx, outWrapper, newInput, newOther, rounding_mode);
}

inline diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDivInp(ctx, inputWrapper, newOther, rounding_mode);
}

inline diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDivScalar(ctx, outWrapper, newInput, other, rounding_mode);
}

inline diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDivInpScalar(ctx, inputWrapper, other, rounding_mode);
}

inline diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    diopiConstTensorHandle_t newInput,newMat2;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mat2, &newMat2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBmm(ctx, outWrapper, newInput, newMat2);
}

inline diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {



    return ::diopiBaddbmm(ctx, out, input, batch1, batch2, beta, alpha);
}

inline diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {



    return ::diopiBaddbmmInp(ctx, input, batch1, batch2, beta, alpha);
}

inline diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcmul(ctx, outWrapper, newInput, newTensor1, newTensor2, value);
}

inline diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcmulInp(ctx, inputWrapper, newTensor1, newTensor2, value);
}

inline diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMatmul(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcdiv(ctx, outWrapper, newInput, newTensor1, newTensor2, value);
}

inline diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcdivInp(ctx, inputWrapper, newTensor1, newTensor2, value);
}

inline diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1, diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput,newMat1,newMat2;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mat1, &newMat1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mat2, &newMat2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddmm(ctx, outWrapper, newInput, newMat1, newMat2, beta, alpha);
}

inline diopiError_t diopiCholesky(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper, bool checkerror) {



    return ::diopiCholesky(ctx, out, info, mat, upper, checkerror);
}

inline diopiError_t diopiCholeskyBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t L, bool upper) {



    return ::diopiCholeskyBackward(ctx, grad_mat, grad_output, L, upper);
}

inline diopiError_t diopiTriangularSolve(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {



    return ::diopiTriangularSolve(ctx, out, cloned_mat, b, mat, upper, transpose, unitriangular);
}

inline diopiError_t diopiTriangularSolveBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat, diopiConstTensorHandle_t x, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {



    return ::diopiTriangularSolveBackward(ctx, grad_b, grad_mat, grad_x, grad_cloned_mat, x, b, mat, upper, transpose, unitriangular);
}

inline diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampInpScalar(ctx, inputWrapper, min, max);
}

inline diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiConstTensorHandle_t newMin,newMax;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, min, &newMin, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, max, &newMax, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampInp(ctx, inputWrapper, newMin, newMax);
}

inline diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampScalar(ctx, outWrapper, newInput, min, max);
}

inline diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiConstTensorHandle_t newInput,newMin,newMax;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, min, &newMin, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, max, &newMax, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClamp(ctx, outWrapper, newInput, newMin, newMax);
}

inline diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {



    return ::diopiClampMaxInpScalar(ctx, input, max);
}

inline diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {



    return ::diopiClampMaxInp(ctx, input, max);
}

inline diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {



    return ::diopiClampMaxScalar(ctx, out, input, max);
}

inline diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {



    return ::diopiClampMax(ctx, out, input, max);
}

inline diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {



    return ::diopiClampMinInpScalar(ctx, input, min);
}

inline diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {



    return ::diopiClampMinInp(ctx, input, min);
}

inline diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {



    return ::diopiClampMinScalar(ctx, out, input, min);
}

inline diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {



    return ::diopiClampMin(ctx, out, input, min);
}

inline diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {



    return ::diopiFill(ctx, input, value);
}

inline diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalAnd(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalAndInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalOr(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalOrInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiLogicalNot(ctx, out, input);
}

inline diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiLogicalNotInp(ctx, input);
}

inline diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiBitwiseAnd(ctx, out, input, other);
}

inline diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiBitwiseAndInp(ctx, input, other);
}

inline diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {



    return ::diopiBitwiseAndScalar(ctx, out, input, other);
}

inline diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {



    return ::diopiBitwiseAndInpScalar(ctx, input, other);
}

inline diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiBitwiseOr(ctx, out, input, other);
}

inline diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiBitwiseOrInp(ctx, input, other);
}

inline diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {



    return ::diopiBitwiseOrScalar(ctx, out, input, other);
}

inline diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {



    return ::diopiBitwiseOrInpScalar(ctx, input, other);
}

inline diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseNot(ctx, outWrapper, newInput);
}

inline diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseNotInp(ctx, inputWrapper);
}

inline diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEqScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEqInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEq(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEqInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNe(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGe(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGtScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGtInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGt(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGtInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLe(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLtScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLtInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLt(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLtInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMean(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSum(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {



    return ::diopiStd(ctx, out, input, dim, unbiased);
}

inline diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto minWrapper = DiopiTensorWrapper<CommonCast>(ctx, min, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto min_indicesWrapper = DiopiTensorWrapper<CommonCast>(ctx, min_indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMin(ctx, minWrapper, min_indicesWrapper, newInput, dim);
}

inline diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {



    return ::diopiMinAll(ctx, min, input);
}

inline diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto maxWrapper = DiopiTensorWrapper<CommonCast>(ctx, max, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto max_indicesWrapper = DiopiTensorWrapper<CommonCast>(ctx, max_indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMax(ctx, maxWrapper, max_indicesWrapper, newInput, dim);
}

inline diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {



    return ::diopiMaxAll(ctx, max, input);
}

inline diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAny(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAll(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSoftmax(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSoftmaxBackward(ctx, grad_inputWrapper, newGrad_output, newOutput, dim);
}

inline diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogSoftmax(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogSoftmaxBackward(ctx, grad_inputWrapper, newGrad_output, newOutput, dim);
}

inline diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {



    return ::diopiIndex(ctx, out, input, indices, nums);
}

inline diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input, diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad) {



    return ::diopiIndexBackward(ctx, grad_input, zeros_like_input, indices, nums, grad);
}

inline diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {



    return ::diopiIndexSelect(ctx, out, input, dim, index);
}

inline diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad, diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {



    return ::diopiIndexSelectBackward(ctx, grad_input, grad, input_sizes, dim, index);
}

inline diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {



    return ::diopiSelect(ctx, out, input, dim, index);
}

inline diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {



    return ::diopiSelectBackward(ctx, grad_input, grad_output, input_sizes, dim, index);
}

inline diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t index) {



    return ::diopiSelectScatter(ctx, out, input, src, dim, index);
}

inline diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t start, int64_t end, int64_t step) {



    return ::diopiSliceScatter(ctx, out, input, src, dim, start, end, step);
}

inline diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end, int64_t step) {



    return ::diopiSlice(ctx, null_out, input, dim, start, end, step);
}

inline diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {



    return ::diopiSliceBackward(ctx, grad_input, grad_output, input_sizes, dim, start, end, step);
}

inline diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {



    return ::diopiMaskedScatter(ctx, out, input, mask, source);
}

inline diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores, double iou_threshold) {



    return ::diopiNms(ctx, out, dets, scores, iou_threshold);
}

inline diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {



    return ::diopiNonzero(ctx, out, input);
}

inline diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLinear(ctx, outWrapper, newInput, newWeight, newBias);
}

inline diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLinearBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight);
}

inline diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio, bool aligned) {



    return ::diopiRoiAlign(ctx, out, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
}

inline diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t batch_size, int64_t channels, int64_t height, int64_t width, int64_t sampling_ratio, bool aligned) {



    return ::diopiRoiAlignBackward(ctx, out, grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned);
}

inline diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {


    auto wWrapper = DiopiTensorWrapper<CommonCast>(ctx, w, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto dwWrapper = DiopiTensorWrapper<CommonCast>(ctx, dw, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto bufWrapper = DiopiTensorWrapper<CommonCast>(ctx, buf, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSgd(ctx, wWrapper, dwWrapper, bufWrapper, lr, momentum, dampening, weight_decay, nesterov);
}

inline diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* grads, int64_t num_grads, double max_norm, double norm_type, bool error_if_nonfinite) {



    return ::diopiClipGradNorm(ctx, out, grads, num_grads, max_norm, norm_type, error_if_nonfinite);
}

inline diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm, double norm_type) {



    return ::diopiEmbeddingRenorm_(ctx, inout, indices, max_norm, norm_type);
}

inline diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {



    return ::diopiEmbedding(ctx, out, weight, indices, padding_idx, scale_grad_byfreq, sparse);
}

inline diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {



    return ::diopiEmbeddingBackward(ctx, out, grad, indices, num_weights, padding_idx, scale_grad_byfreq, sparse);
}

inline diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {



    return ::diopiTril(ctx, out, input, diagonal);
}

inline diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim) {



    return ::diopiCat(ctx, out, tensors, num_inputs, dim);
}

inline diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs, diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {



    return ::diopiSplitWithSizes(ctx, outs, num_outs, input, splitSizes, dim);
}

inline diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    std::vector<diopiConstTensorHandle_t> newTensors(numTensors, diopiConstTensorHandle_t());
    for (int i = 0; i < numTensors; ++i) {
        castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, tensors[i], &newTensors[i], std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    }

    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiStack(ctx, outWrapper, newTensors.data(), numTensors, dim);
}

inline diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto valuesWrapper = DiopiTensorWrapper<CommonCast>(ctx, values, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<CommonCast>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSort(ctx, valuesWrapper, indicesWrapper, newInput, dim, descending, stable);
}

inline diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto valuesWrapper = DiopiTensorWrapper<CommonCast>(ctx, values, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<CommonCast>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTopk(ctx, valuesWrapper, indicesWrapper, newInput, k, dim, largest, sorted);
}

inline diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTranspose(ctx, outWrapper, newInput, dim0, dim1);
}

inline diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiOneHot(ctx, outWrapper, newInput, num_classes);
}

inline diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newCondition,newInput,newOther;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, condition, &newCondition, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiWhere(ctx, outWrapper, newCondition, newInput, newOther);
}

inline diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    diopiConstTensorHandle_t newInput,newMask,newValue;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, value, &newValue, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFill(ctx, outWrapper, newInput, newMask, newValue);
}

inline diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    diopiConstTensorHandle_t newMask,newValue;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, value, &newValue, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFillInp(ctx, inputWrapper, newMask, newValue);
}

inline diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newMask;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFillScalar(ctx, outWrapper, newInput, newMask, value);
}

inline diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newMask;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFillInpScalar(ctx, inputWrapper, newMask, value);
}

inline diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiReciprocal(ctx, outWrapper, newInput);
}

inline diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiReciprocalInp(ctx, inputWrapper);
}

inline diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {


    auto inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto gradWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto exp_avgWrapper = DiopiTensorWrapper<CommonCast>(ctx, exp_avg, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto exp_avg_sqWrapper = DiopiTensorWrapper<CommonCast>(ctx, exp_avg_sq, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto max_exp_avg_sqWrapper = DiopiTensorWrapper<CommonCast>(ctx, max_exp_avg_sq, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdamW(ctx, inputWrapper, gradWrapper, exp_avgWrapper, exp_avg_sqWrapper, max_exp_avg_sqWrapper, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

inline diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {



    return ::diopiConvTranspose2d(ctx, out, input, weight, bias, stride, padding, output_padding, groups, dilation);
}

inline diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUnfold(ctx, outWrapper, newInput, dim, size, step);
}

inline diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
    diopiConstTensorHandle_t newGrad_output;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUnfoldBackward(ctx, grad_inputWrapper, newGrad_output, input_sizes, dim, size, step);
}

inline diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCumsum(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, const int64_t* compute_mode) {
    diopiConstTensorHandle_t newInput1,newInput2;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input1, &newInput1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input2, &newInput2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCdist(ctx, outWrapper, newInput1, newInput2, p, compute_mode);
}

inline diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    diopiConstTensorHandle_t newGrad_output,newInput1,newInput2,newCdist;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input1, &newInput1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input2, &newInput2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, cdist, &newCdist, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCdistBackward(ctx, grad_inputWrapper, newGrad_output, newInput1, newInput2, p, newCdist);
}

inline diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {



    return ::diopiArgmax(ctx, out, input, dim, keepdim);
}

inline diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {



    return ::diopiAdadelta(ctx, input, grad, square_avg, acc_delta, lr, rho, eps, weight_decay);
}

inline diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {



    return ::diopiAdam(ctx, input, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

inline diopiError_t diopiRmsprop(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t grad_avg, diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered) {



    return ::diopiRmsprop(ctx, input, grad, square_avg, grad_avg, momentum_buf, lr, alpha, eps, weight_decay, momentum, centered);
}

inline diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {



    return ::diopiSmoothL1Loss(ctx, out, input, target, reduction, beta);
}

inline diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {



    return ::diopiSmoothL1LossBackward(ctx, grad_input, grad_output, input, target, reduction, beta);
}

inline diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {



    return ::diopiConvolution3d(ctx, out, input, weight, bias, stride, padding, dilation, groups);
}

inline diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {



    return ::diopiConvolution3dBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
}

inline diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {



    return ::diopiMaxPool3d(ctx, out, input, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {



    return ::diopiMaxPool3dWithIndices(ctx, out, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {



    return ::diopiMaxPool3dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}

inline diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {



    return ::diopiAdaptiveAvgPool3d(ctx, out, input, output_size);
}

inline diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {



    return ::diopiAdaptiveAvgPool3dBackward(ctx, grad_input, grad_output, input);
}

inline diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {



    return ::diopiAdaptiveMaxPool3d(ctx, out, input, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {



    return ::diopiAdaptiveMaxPool3dWithIndices(ctx, out, indices, input, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {



    return ::diopiAdaptiveMaxPool3dBackward(ctx, grad_input, grad_output, input, indices);
}

inline diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {



    return ::diopiMaskedSelect(ctx, out, input, mask);
}

inline diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {



    return ::diopiMaskedSelectBackward(ctx, grad_input, grad_output, input, mask);
}

inline diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiMaximum(ctx, out, input, other);
}

inline diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiMinimum(ctx, out, input, other);
}

inline diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {



    return ::diopiMm(ctx, out, input, mat2);
}

inline diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {



    return ::diopiIndexFillScalar(ctx, out, input, dim, index, value);
}

inline diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {



    return ::diopiIndexFill(ctx, out, input, dim, index, value);
}

inline diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {



    return ::diopiIndexFillInpScalar(ctx, input, dim, index, value);
}

inline diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {



    return ::diopiIndexFillInp(ctx, input, dim, index, value);
}

inline diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiExpand(ctx, out, input);
}

inline diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {



    return ::diopiLinspace(ctx, out, start, end, steps);
}

inline diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPermute(ctx, outWrapper, newInput, dims);
}

inline diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode, double* value) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPad(ctx, outWrapper, newInput, pad, mode, value);
}

inline diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRoll(ctx, outWrapper, newInput, shifts, dims);
}

inline diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {



    return ::diopiFlip(ctx, out, input, dims);
}

inline diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNorm(ctx, outWrapper, newInput, p, dim);
}

inline diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {



    return ::diopiGroupNorm(ctx, out, save_mean, save_invstd, input, weight, bias, num_groups, eps);
}

inline diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, int64_t num_groups) {



    return ::diopiGroupNormBackward(ctx, grad_input, grad_weight, grad_bias, grad_output, input, weight, mean, rstd, num_groups);
}

inline diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted, bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {



    return ::diopiUnique(ctx, out, input, dim, sorted, return_counts, indices, counts);
}

inline diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiProd(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {



    return ::diopiCTCLoss(ctx, out, neg_log_likelihood, log_alpha, log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
}

inline diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {



    return ::diopiCTCLossBackward(ctx, grad_input, grad_output, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, reduction, zero_infinity);
}

inline diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {



    return ::diopiRemainderTensor(ctx, out, input, other);
}

inline diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {



    return ::diopiRemainderScalar(ctx, out, input, other);
}

inline diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {



    return ::diopiRemainder(ctx, out, input, other);
}

inline diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {



    return ::diopiGather(ctx, out, input, dim, index);
}

inline diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {



    return ::diopiGatherBackward(ctx, grad_input, grad_output, input, dim, index);
}

inline diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {



    return ::diopiScatterInp(ctx, input, dim, src, index, reduce);
}

inline diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {



    return ::diopiScatterInpScalar(ctx, input, dim, value, index, reduce);
}

inline diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {



    return ::diopiScatter(ctx, out, input, dim, src, index, reduce);
}

inline diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {



    return ::diopiScatterScalar(ctx, out, input, dim, value, index, reduce);
}

inline diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate) {



    return ::diopiIndexPutInp(ctx, input, values, indices, indices_counts, accumulate);
}

inline diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate) {



    return ::diopiIndexPut(ctx, out, input, values, indices, indices_counts, accumulate);
}

inline diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {


    auto inoutWrapper = DiopiTensorWrapper<CommonCast>(ctx, inout, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRandomInp(ctx, inoutWrapper, from, to, idx);
}

inline diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {



    return ::diopiUniformInp(ctx, inout, from, to, idx);
}

inline diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx) {



    return ::diopiBernoulli(ctx, out, input, idx);
}

inline diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {



    return ::diopiBernoulliInp(ctx, inout, idx);
}

inline diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {



    return ::diopiBernoulliScalar(ctx, out, p, idx);
}

inline diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {


    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiArange(ctx, outWrapper, start, end, step);
}

inline diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {


    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRandperm(ctx, outWrapper, n, idx);
}

inline diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std) {



    return ::diopiNormal(ctx, out, mean, std);
}

inline diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std) {



    return ::diopiNormalTensorScalar(ctx, out, mean, std);
}

inline diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std) {



    return ::diopiNormalScalarTensor(ctx, out, mean, std);
}

inline diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std) {



    return ::diopiNormalTensor(ctx, out, mean, std);
}

inline diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std) {



    return ::diopiNormalInp(ctx, inout, mean, std);
}

inline diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum) {



    return ::diopiMeshGrid(ctx, outs, inputs, inputsNum);
}

inline diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement) {



    return ::diopiMultinomial(ctx, out, input, num_samples, replacement);
}

inline diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalized_shape, double eps) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_meanWrapper = DiopiTensorWrapper<CommonCast>(ctx, save_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_invstdWrapper = DiopiTensorWrapper<CommonCast>(ctx, save_invstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLayerNorm(ctx, outWrapper, save_meanWrapper, save_invstdWrapper, newInput, newWeight, newBias, normalized_shape, eps);
}

inline diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight,newBias,newMean,newRstd;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, mean, &newMean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, rstd, &newRstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<CommonCast>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLayerNormBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, newBias, newMean, newRstd, normalized_shape);
}

inline diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {



    return ::diopiCopyInp(ctx, src, input);
}

inline diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {



    return ::diopiUpsampleNearest(ctx, out, input, size);
}

inline diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size) {



    return ::diopiUpsampleNearestBackward(ctx, grad_input, grad_output, out_size, in_size);
}

inline diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool align_corners, const char* mode) {



    return ::diopiUpsampleLinear(ctx, out, input, size, align_corners, mode);
}

inline diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode) {



    return ::diopiUpsampleLinearBackward(ctx, grad_input, grad_output, out_size, in_size, align_corners, mode);
}

inline diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiErfinv(ctx, out, input);
}

inline diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {



    return ::diopiErfinvInp(ctx, input);
}

inline diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {



    return ::diopiIm2Col(ctx, out, input, kernel_size, dilation, padding, stride);
}

inline diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {



    return ::diopiCol2Im(ctx, out, input, output_size, kernel_size, dilation, padding, stride);
}

inline diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t,CommonCast>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<CommonCast>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRepeat(ctx, outWrapper, newInput, repeats_size);
}

inline diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {



    return ::diopiCastDtype(ctx, out, input);
}

}
# endif // DIOPI_ADAPTOR_HPP
