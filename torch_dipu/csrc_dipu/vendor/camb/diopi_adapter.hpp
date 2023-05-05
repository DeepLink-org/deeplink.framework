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

class diopiAddCast {
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
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiConvolution2d(ctx, outWrapper, newInput, newWeight, newBias, stride, padding, dilation, groups);
}

inline diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiConvolution2dBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
}

inline diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean, diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_meanWrapper = DiopiTensorWrapper<>(ctx, save_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_invstdWrapper = DiopiTensorWrapper<>(ctx, save_invstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto running_meanWrapper = DiopiTensorWrapper<>(ctx, running_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto running_varWrapper = DiopiTensorWrapper<>(ctx, running_var, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBatchNorm(ctx, outWrapper, save_meanWrapper, save_invstdWrapper, newInput, newWeight, newBias, running_meanWrapper, running_varWrapper, training, momentum, eps);
}

inline diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean, diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight,newRunning_mean,newRunning_var,newSave_mean,newSave_invstd;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, running_mean, &newRunning_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, running_var, &newRunning_var, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, save_mean, &newSave_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, save_invstd, &newSave_invstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBatchNormBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, newRunning_mean, newRunning_var, newSave_mean, newSave_invstd, training, eps);
}

inline diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRelu(ctx, outWrapper, newInput);
}

inline diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiReluInp(ctx, inputWrapper);
}

inline diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardtanh(ctx, outWrapper, newInput, min_val, max_val);
}

inline diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardtanhInp(ctx, inputWrapper, min_val, max_val);
}

inline diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardtanhBackward(ctx, grad_inputWrapper, newGrad_output, newInput, min_val, max_val);
}

inline diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardswish(ctx, outWrapper, newInput);
}

inline diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardswishInp(ctx, inputWrapper);
}

inline diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiHardswishBackward(ctx, grad_inputWrapper, newGrad_output, newInput);
}

inline diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiThreshold(ctx, outWrapper, newInput, threshold, value);
}

inline diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiThresholdInp(ctx, inputWrapper, threshold, value);
}

inline diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiThresholdBackward(ctx, grad_inputWrapper, newGrad_output, newInput, threshold);
}

inline diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGelu(ctx, outWrapper, newInput, approximate);
}

inline diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const char* approximate) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeluBackward(ctx, grad_inputWrapper, newGrad_output, newInput, approximate);
}

inline diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeakyRelu(ctx, outWrapper, newInput, negative_slope);
}

inline diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeakyReluInp(ctx, inputWrapper, negative_slope);
}

inline diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeakyReluBackward(ctx, grad_inputWrapper, newGrad_output, newInput, negative_slope, input_is_result);
}

inline diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAvgPool2d(ctx, outWrapper, newInput, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

inline diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAvgPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
}

inline diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool2d(ctx, outWrapper, newInput, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool2dWithIndices(ctx, outWrapper, indicesWrapper, newInput, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    diopiConstTensorHandle_t newGrad_output,newInput,newIndices;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, kernel_size, stride, padding, dilation, ceil_mode, newIndices);
}

inline diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveAvgPool2d(ctx, outWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveAvgPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput);
}

inline diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool2d(ctx, outWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool2dWithIndices(ctx, outWrapper, indicesWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    diopiConstTensorHandle_t newGrad_output,newInput,newIndices;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool2dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newIndices);
}

inline diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto maskWrapper = DiopiTensorWrapper<>(ctx, mask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDropout(ctx, outWrapper, maskWrapper, newInput, p, train);
}

inline diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto maskWrapper = DiopiTensorWrapper<>(ctx, mask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDropoutInp(ctx, inputWrapper, maskWrapper, p, train);
}

inline diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newInput,newTarget;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMSELoss(ctx, outWrapper, newInput, newTarget, reduction);
}

inline diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMSELossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, reduction);
}

inline diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs, diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newInputs,newTargets;
    castImpl<diopiConstTensorHandle_t>(ctx, inputs, &newInputs, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, targets, &newTargets, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoidFocalLoss(ctx, outWrapper, newInputs, newTargets, alpha, gamma, reduction);
}

inline diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newInput,newTarget;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_outputWrapper = DiopiTensorWrapper<>(ctx, grad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoidFocalLossBackward(ctx, grad_outputWrapper, newInput, newTarget, grad_inputWrapper, gamma, alpha, reduction);
}

inline diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    diopiConstTensorHandle_t newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCrossEntropyLoss(ctx, outWrapper, newInput, newTarget, newWeight, reduction, ignore_index, label_smoothing);
}

inline diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCrossEntropyLossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, newWeight, reduction, ignore_index, label_smoothing);
}

inline diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    diopiConstTensorHandle_t newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNLLLoss(ctx, outWrapper, newInput, newTarget, newWeight, reduction, ignore_index);
}

inline diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNLLLossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, newWeight, reduction, ignore_index);
}

inline diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newInput,newTarget,newWeight,newPos_weight;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, pos_weight, &newPos_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBCEWithLogits(ctx, outWrapper, newInput, newTarget, newWeight, newPos_weight, reduction);
}

inline diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget,newWeight,newPos_weight;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, pos_weight, &newPos_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBCEWithLogitsBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, newWeight, newPos_weight, reduction);
}

inline diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBCELoss(ctx, outWrapper, newInput, newTarget, newWeight, reduction);
}

inline diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBCELossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, newWeight, reduction);
}

inline diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSign(ctx, outWrapper, newInput);
}

inline diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAbsInp(ctx, inputWrapper);
}

inline diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAbs(ctx, outWrapper, newInput);
}

inline diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNegInp(ctx, inputWrapper);
}

inline diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeg(ctx, outWrapper, newInput);
}

inline diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiFloorInp(ctx, inputWrapper);
}

inline diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiFloor(ctx, outWrapper, newInput);
}

inline diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSqrtInp(ctx, inputWrapper);
}

inline diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSqrt(ctx, outWrapper, newInput);
}

inline diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRsqrtInp(ctx, inputWrapper);
}

inline diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRsqrt(ctx, outWrapper, newInput);
}

inline diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSinInp(ctx, inputWrapper);
}

inline diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSin(ctx, outWrapper, newInput);
}

inline diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCosInp(ctx, inputWrapper);
}

inline diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCos(ctx, outWrapper, newInput);
}

inline diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTanhInp(ctx, inputWrapper);
}

inline diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTanh(ctx, outWrapper, newInput);
}

inline diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTanhBackward(ctx, grad_inputWrapper, newGrad_output, newOutput);
}

inline diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoidInp(ctx, inputWrapper);
}

inline diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoid(ctx, outWrapper, newInput);
}

inline diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSigmoidBackward(ctx, grad_inputWrapper, newGrad_output, newOutput);
}

inline diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSiluInp(ctx, inputWrapper);
}

inline diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSilu(ctx, outWrapper, newInput);
}

inline diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSiluBackward(ctx, grad_inputWrapper, newGrad_output, newInput);
}

inline diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiExpInp(ctx, inputWrapper);
}

inline diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiExp(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogInp(ctx, inputWrapper);
}

inline diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog2Inp(ctx, inputWrapper);
}

inline diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog2(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog10Inp(ctx, inputWrapper);
}

inline diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLog10(ctx, outWrapper, newInput);
}

inline diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiErfInp(ctx, inputWrapper);
}

inline diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiErf(ctx, outWrapper, newInput);
}

inline diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    diopiConstTensorHandle_t newExponent;
    castImpl<diopiConstTensorHandle_t>(ctx, exponent, &newExponent, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPowScalar(ctx, outWrapper, input, newExponent);
}

inline diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPow(ctx, outWrapper, newInput, exponent);
}

inline diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPowInp(ctx, inputWrapper, exponent);
}

inline diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    diopiConstTensorHandle_t newInput,newExponent;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, exponent, &newExponent, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPowTensor(ctx, outWrapper, newInput, newExponent);
}

inline diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    diopiConstTensorHandle_t newExponent;
    castImpl<diopiConstTensorHandle_t>(ctx, exponent, &newExponent, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPowInpTensor(ctx, inputWrapper, newExponent);
}

inline diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t, diopiAddInputCast>(ctx, input, &newInput);
    castImpl<diopiConstTensorHandle_t, diopiAddOtherCast>(ctx, other, &newOther);
    auto outWrapper = DiopiTensorWrapper<diopiAddOtherCast, true>(ctx, out);
    return ::diopiAdd(ctx, outWrapper, newInput, newOther, alpha);
}

inline diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddInp(ctx, inputWrapper, newOther, alpha);
}

inline diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddScalar(ctx, outWrapper, newInput, other, alpha);
}

inline diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddInpScalar(ctx, inputWrapper, other, alpha);
}

inline diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSub(ctx, outWrapper, newInput, newOther, alpha);
}

inline diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSubInp(ctx, inputWrapper, newOther, alpha);
}

inline diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSubScalar(ctx, outWrapper, newInput, other, alpha);
}

inline diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSubInpScalar(ctx, inputWrapper, other, alpha);
}

inline diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMul(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMulInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMulScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMulInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDiv(ctx, outWrapper, newInput, newOther, rounding_mode);
}

inline diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDivInp(ctx, inputWrapper, newOther, rounding_mode);
}

inline diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDivScalar(ctx, outWrapper, newInput, other, rounding_mode);
}

inline diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiDivInpScalar(ctx, inputWrapper, other, rounding_mode);
}

inline diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    diopiConstTensorHandle_t newInput,newMat2;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mat2, &newMat2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBmm(ctx, outWrapper, newInput, newMat2);
}

inline diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
    diopiConstTensorHandle_t newInput,newBatch1,newBatch2;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, batch1, &newBatch1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, batch2, &newBatch2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBaddbmm(ctx, outWrapper, newInput, newBatch1, newBatch2, beta, alpha);
}

inline diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta, double alpha) {
    diopiConstTensorHandle_t newBatch1,newBatch2;
    castImpl<diopiConstTensorHandle_t>(ctx, batch1, &newBatch1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, batch2, &newBatch2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBaddbmmInp(ctx, inputWrapper, newBatch1, newBatch2, beta, alpha);
}

inline diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcmul(ctx, outWrapper, newInput, newTensor1, newTensor2, value);
}

inline diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcmulInp(ctx, inputWrapper, newTensor1, newTensor2, value);
}

inline diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMatmul(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcdiv(ctx, outWrapper, newInput, newTensor1, newTensor2, value);
}

inline diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newTensor1,newTensor2;
    castImpl<diopiConstTensorHandle_t>(ctx, tensor1, &newTensor1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, tensor2, &newTensor2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddcdivInp(ctx, inputWrapper, newTensor1, newTensor2, value);
}

inline diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1, diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    diopiConstTensorHandle_t newInput,newMat1,newMat2;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mat1, &newMat1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mat2, &newMat2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAddmm(ctx, outWrapper, newInput, newMat1, newMat2, beta, alpha);
}

inline diopiError_t diopiCholesky(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper, bool checkerror) {
    diopiConstTensorHandle_t newMat;
    castImpl<diopiConstTensorHandle_t>(ctx, mat, &newMat, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto infoWrapper = DiopiTensorWrapper<>(ctx, info, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCholesky(ctx, outWrapper, infoWrapper, newMat, upper, checkerror);
}

inline diopiError_t diopiCholeskyBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t L, bool upper) {
    diopiConstTensorHandle_t newGrad_output,newL;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, L, &newL, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_matWrapper = DiopiTensorWrapper<>(ctx, grad_mat, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCholeskyBackward(ctx, grad_matWrapper, newGrad_output, newL, upper);
}

inline diopiError_t diopiTriangularSolve(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
    diopiConstTensorHandle_t newB,newMat;
    castImpl<diopiConstTensorHandle_t>(ctx, b, &newB, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mat, &newMat, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto cloned_matWrapper = DiopiTensorWrapper<>(ctx, cloned_mat, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTriangularSolve(ctx, outWrapper, cloned_matWrapper, newB, newMat, upper, transpose, unitriangular);
}

inline diopiError_t diopiTriangularSolveBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat, diopiConstTensorHandle_t x, diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
    diopiConstTensorHandle_t newGrad_x,newGrad_cloned_mat,newX,newB,newMat;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_x, &newGrad_x, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, grad_cloned_mat, &newGrad_cloned_mat, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, x, &newX, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, b, &newB, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mat, &newMat, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_bWrapper = DiopiTensorWrapper<>(ctx, grad_b, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_matWrapper = DiopiTensorWrapper<>(ctx, grad_mat, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTriangularSolveBackward(ctx, grad_bWrapper, grad_matWrapper, newGrad_x, newGrad_cloned_mat, newX, newB, newMat, upper, transpose, unitriangular);
}

inline diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampInpScalar(ctx, inputWrapper, min, max);
}

inline diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiConstTensorHandle_t newMin,newMax;
    castImpl<diopiConstTensorHandle_t>(ctx, min, &newMin, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, max, &newMax, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampInp(ctx, inputWrapper, newMin, newMax);
}

inline diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampScalar(ctx, outWrapper, newInput, min, max);
}

inline diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    diopiConstTensorHandle_t newInput,newMin,newMax;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, min, &newMin, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, max, &newMax, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClamp(ctx, outWrapper, newInput, newMin, newMax);
}

inline diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMaxInpScalar(ctx, inputWrapper, max);
}

inline diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    diopiConstTensorHandle_t newMax;
    castImpl<diopiConstTensorHandle_t>(ctx, max, &newMax, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMaxInp(ctx, inputWrapper, newMax);
}

inline diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMaxScalar(ctx, outWrapper, newInput, max);
}

inline diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    diopiConstTensorHandle_t newInput,newMax;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, max, &newMax, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMax(ctx, outWrapper, newInput, newMax);
}

inline diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMinInpScalar(ctx, inputWrapper, min);
}

inline diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    diopiConstTensorHandle_t newMin;
    castImpl<diopiConstTensorHandle_t>(ctx, min, &newMin, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMinInp(ctx, inputWrapper, newMin);
}

inline diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMinScalar(ctx, outWrapper, newInput, min);
}

inline diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    diopiConstTensorHandle_t newInput,newMin;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, min, &newMin, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiClampMin(ctx, outWrapper, newInput, newMin);
}

inline diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiFill(ctx, inputWrapper, value);
}

inline diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalAnd(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalAndInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalOr(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalOrInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalNot(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogicalNotInp(ctx, inputWrapper);
}

inline diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseAnd(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseAndInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseAndScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseAndInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseOr(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseOrInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseOrScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseOrInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseNot(ctx, outWrapper, newInput);
}

inline diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBitwiseNotInp(ctx, inputWrapper);
}

inline diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEqScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEqInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEq(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEqInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNe(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNeInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGe(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGeInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGtScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGtInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGt(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGtInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLe(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLeInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLtScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLtInpScalar(ctx, inputWrapper, other);
}

inline diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLt(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLtInp(ctx, inputWrapper, newOther);
}

inline diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMean(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSum(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiStd(ctx, outWrapper, newInput, dim, unbiased);
}

inline diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto minWrapper = DiopiTensorWrapper<>(ctx, min, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto min_indicesWrapper = DiopiTensorWrapper<>(ctx, min_indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMin(ctx, minWrapper, min_indicesWrapper, newInput, dim);
}

inline diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto minWrapper = DiopiTensorWrapper<>(ctx, min, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMinAll(ctx, minWrapper, newInput);
}

inline diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto maxWrapper = DiopiTensorWrapper<>(ctx, max, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto max_indicesWrapper = DiopiTensorWrapper<>(ctx, max_indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMax(ctx, maxWrapper, max_indicesWrapper, newInput, dim);
}

inline diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto maxWrapper = DiopiTensorWrapper<>(ctx, max, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxAll(ctx, maxWrapper, newInput);
}

inline diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAny(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAll(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSoftmax(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSoftmaxBackward(ctx, grad_inputWrapper, newGrad_output, newOutput, dim);
}

inline diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogSoftmax(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output, int64_t dim) {
    diopiConstTensorHandle_t newGrad_output,newOutput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, output, &newOutput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLogSoftmaxBackward(ctx, grad_inputWrapper, newGrad_output, newOutput, dim);
}

inline diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {



    return ::diopiIndex(ctx, out, input, indices, nums);
}

inline diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input, diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad) {
    std::vector<diopiConstTensorHandle_t> newIndices(nums, diopiConstTensorHandle_t());
    for (int i = 0; i < nums; ++i) {
        castImpl<diopiConstTensorHandle_t>(ctx, indices[i], &newIndices[i], std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    }
    diopiConstTensorHandle_t newGrad;
    castImpl<diopiConstTensorHandle_t>(ctx, grad, &newGrad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto zeros_like_inputWrapper = DiopiTensorWrapper<>(ctx, zeros_like_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIndexBackward(ctx, grad_inputWrapper, zeros_like_inputWrapper, newIndices.data(), nums, newGrad);
}

inline diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiConstTensorHandle_t newInput,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIndexSelect(ctx, outWrapper, newInput, dim, newIndex);
}

inline diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad, diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {
    diopiConstTensorHandle_t newGrad,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, grad, &newGrad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIndexSelectBackward(ctx, grad_inputWrapper, newGrad, input_sizes, dim, newIndex);
}

inline diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSelect(ctx, outWrapper, newInput, dim, index);
}

inline diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
    diopiConstTensorHandle_t newGrad_output;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSelectBackward(ctx, grad_inputWrapper, newGrad_output, input_sizes, dim, index);
}

inline diopiError_t diopiSelectScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t index) {
    diopiConstTensorHandle_t newInput,newSrc;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, src, &newSrc, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSelectScatter(ctx, outWrapper, newInput, newSrc, dim, index);
}

inline diopiError_t diopiSliceScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t src, int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiConstTensorHandle_t newInput,newSrc;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, src, &newSrc, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSliceScatter(ctx, outWrapper, newInput, newSrc, dim, start, end, step);
}

inline diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto null_outWrapper = DiopiTensorWrapper<>(ctx, null_out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSlice(ctx, null_outWrapper, newInput, dim, start, end, step);
}

inline diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    diopiConstTensorHandle_t newGrad_output;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSliceBackward(ctx, grad_inputWrapper, newGrad_output, input_sizes, dim, start, end, step);
}

inline diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {
    diopiConstTensorHandle_t newInput,newMask,newSource;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, source, &newSource, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedScatter(ctx, outWrapper, newInput, newMask, newSource);
}

inline diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores, double iou_threshold) {



    return ::diopiNms(ctx, out, dets, scores, iou_threshold);
}

inline diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {



    return ::diopiNonzero(ctx, out, input);
}

inline diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLinear(ctx, outWrapper, newInput, newWeight, newBias);
}

inline diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLinearBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight);
}

inline diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sampling_ratio, bool aligned) {
    diopiConstTensorHandle_t newInput,newRois;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, rois, &newRois, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRoiAlign(ctx, outWrapper, newInput, newRois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned);
}

inline diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t batch_size, int64_t channels, int64_t height, int64_t width, int64_t sampling_ratio, bool aligned) {
    diopiConstTensorHandle_t newGrad,newRois;
    castImpl<diopiConstTensorHandle_t>(ctx, grad, &newGrad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, rois, &newRois, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRoiAlignBackward(ctx, outWrapper, newGrad, newRois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned);
}

inline diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double lr, double momentum, double dampening, double weight_decay, bool nesterov) {


    auto wWrapper = DiopiTensorWrapper<>(ctx, w, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto dwWrapper = DiopiTensorWrapper<>(ctx, dw, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto bufWrapper = DiopiTensorWrapper<>(ctx, buf, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSgd(ctx, wWrapper, dwWrapper, bufWrapper, lr, momentum, dampening, weight_decay, nesterov);
}

inline diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* grads, int64_t num_grads, double max_norm, double norm_type, bool error_if_nonfinite) {



    return ::diopiClipGradNorm(ctx, out, grads, num_grads, max_norm, norm_type, error_if_nonfinite);
}

inline diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
    diopiConstTensorHandle_t newIndices;
    castImpl<diopiConstTensorHandle_t>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inoutWrapper = DiopiTensorWrapper<>(ctx, inout, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEmbeddingRenorm_(ctx, inoutWrapper, newIndices, max_norm, norm_type);
}

inline diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    diopiConstTensorHandle_t newWeight,newIndices;
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEmbedding(ctx, outWrapper, newWeight, newIndices, padding_idx, scale_grad_byfreq, sparse);
}

inline diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    diopiConstTensorHandle_t newGrad,newIndices;
    castImpl<diopiConstTensorHandle_t>(ctx, grad, &newGrad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiEmbeddingBackward(ctx, outWrapper, newGrad, newIndices, num_weights, padding_idx, scale_grad_byfreq, sparse);
}

inline diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTril(ctx, outWrapper, newInput, diagonal);
}

inline diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim) {
    std::vector<diopiConstTensorHandle_t> newTensors(num_inputs, diopiConstTensorHandle_t());
    for (int i = 0; i < num_inputs; ++i) {
        castImpl<diopiConstTensorHandle_t>(ctx, tensors[i], &newTensors[i], std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    }

    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCat(ctx, outWrapper, newTensors.data(), num_inputs, dim);
}

inline diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t num_outs, diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {



    return ::diopiSplitWithSizes(ctx, outs, num_outs, input, splitSizes, dim);
}

inline diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    std::vector<diopiConstTensorHandle_t> newTensors(numTensors, diopiConstTensorHandle_t());
    for (int i = 0; i < numTensors; ++i) {
        castImpl<diopiConstTensorHandle_t>(ctx, tensors[i], &newTensors[i], std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    }

    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiStack(ctx, outWrapper, newTensors.data(), numTensors, dim);
}

inline diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto valuesWrapper = DiopiTensorWrapper<>(ctx, values, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSort(ctx, valuesWrapper, indicesWrapper, newInput, dim, descending, stable);
}

inline diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto valuesWrapper = DiopiTensorWrapper<>(ctx, values, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTopk(ctx, valuesWrapper, indicesWrapper, newInput, k, dim, largest, sorted);
}

inline diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiTranspose(ctx, outWrapper, newInput, dim0, dim1);
}

inline diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_classes) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiOneHot(ctx, outWrapper, newInput, num_classes);
}

inline diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newCondition,newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, condition, &newCondition, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiWhere(ctx, outWrapper, newCondition, newInput, newOther);
}

inline diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    diopiConstTensorHandle_t newInput,newMask,newValue;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, value, &newValue, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFill(ctx, outWrapper, newInput, newMask, newValue);
}

inline diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    diopiConstTensorHandle_t newMask,newValue;
    castImpl<diopiConstTensorHandle_t>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, value, &newValue, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFillInp(ctx, inputWrapper, newMask, newValue);
}

inline diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newMask;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFillScalar(ctx, outWrapper, newInput, newMask, value);
}

inline diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newMask;
    castImpl<diopiConstTensorHandle_t>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedFillInpScalar(ctx, inputWrapper, newMask, value);
}

inline diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiReciprocal(ctx, outWrapper, newInput);
}

inline diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiReciprocalInp(ctx, inputWrapper);
}

inline diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto gradWrapper = DiopiTensorWrapper<>(ctx, grad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto exp_avgWrapper = DiopiTensorWrapper<>(ctx, exp_avg, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto exp_avg_sqWrapper = DiopiTensorWrapper<>(ctx, exp_avg_sq, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto max_exp_avg_sqWrapper = DiopiTensorWrapper<>(ctx, max_exp_avg_sq, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdamW(ctx, inputWrapper, gradWrapper, exp_avgWrapper, exp_avg_sqWrapper, max_exp_avg_sqWrapper, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

inline diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiConvTranspose2d(ctx, outWrapper, newInput, newWeight, newBias, stride, padding, output_padding, groups, dilation);
}

inline diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUnfold(ctx, outWrapper, newInput, dim, size, step);
}

inline diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
    diopiConstTensorHandle_t newGrad_output;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUnfoldBackward(ctx, grad_inputWrapper, newGrad_output, input_sizes, dim, size, step);
}

inline diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCumsum(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, const int64_t* compute_mode) {
    diopiConstTensorHandle_t newInput1,newInput2;
    castImpl<diopiConstTensorHandle_t>(ctx, input1, &newInput1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input2, &newInput2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCdist(ctx, outWrapper, newInput1, newInput2, p, compute_mode);
}

inline diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    diopiConstTensorHandle_t newGrad_output,newInput1,newInput2,newCdist;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input1, &newInput1, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input2, &newInput2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, cdist, &newCdist, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCdistBackward(ctx, grad_inputWrapper, newGrad_output, newInput1, newInput2, p, newCdist);
}

inline diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiArgmax(ctx, outWrapper, newInput, dim, keepdim);
}

inline diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto gradWrapper = DiopiTensorWrapper<>(ctx, grad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto square_avgWrapper = DiopiTensorWrapper<>(ctx, square_avg, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto acc_deltaWrapper = DiopiTensorWrapper<>(ctx, acc_delta, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdadelta(ctx, inputWrapper, gradWrapper, square_avgWrapper, acc_deltaWrapper, lr, rho, eps, weight_decay);
}

inline diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto gradWrapper = DiopiTensorWrapper<>(ctx, grad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto exp_avgWrapper = DiopiTensorWrapper<>(ctx, exp_avg, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto exp_avg_sqWrapper = DiopiTensorWrapper<>(ctx, exp_avg_sq, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto max_exp_avg_sqWrapper = DiopiTensorWrapper<>(ctx, max_exp_avg_sq, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdam(ctx, inputWrapper, gradWrapper, exp_avgWrapper, exp_avg_sqWrapper, max_exp_avg_sqWrapper, lr, beta1, beta2, eps, weight_decay, step, amsgrad);
}

inline diopiError_t diopiRmsprop(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg, diopiTensorHandle_t grad_avg, diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto gradWrapper = DiopiTensorWrapper<>(ctx, grad, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto square_avgWrapper = DiopiTensorWrapper<>(ctx, square_avg, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_avgWrapper = DiopiTensorWrapper<>(ctx, grad_avg, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto momentum_bufWrapper = DiopiTensorWrapper<>(ctx, momentum_buf, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRmsprop(ctx, inputWrapper, gradWrapper, square_avgWrapper, grad_avgWrapper, momentum_bufWrapper, lr, alpha, eps, weight_decay, momentum, centered);
}

inline diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
    diopiConstTensorHandle_t newInput,newTarget;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSmoothL1Loss(ctx, outWrapper, newInput, newTarget, reduction, beta);
}

inline diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
    diopiConstTensorHandle_t newGrad_output,newInput,newTarget;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target, &newTarget, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiSmoothL1LossBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newTarget, reduction, beta);
}

inline diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiConvolution3d(ctx, outWrapper, newInput, newWeight, newBias, stride, padding, dilation, groups);
}

inline diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiConvolution3dBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups);
}

inline diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool3d(ctx, outWrapper, newInput, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool3dWithIndices(ctx, outWrapper, indicesWrapper, newInput, kernel_size, stride, padding, dilation, ceil_mode);
}

inline diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    diopiConstTensorHandle_t newGrad_output,newInput,newIndices;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaxPool3dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, kernel_size, stride, padding, dilation, ceil_mode, newIndices);
}

inline diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveAvgPool3d(ctx, outWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newGrad_output,newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveAvgPool3dBackward(ctx, grad_inputWrapper, newGrad_output, newInput);
}

inline diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool3d(ctx, outWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto indicesWrapper = DiopiTensorWrapper<>(ctx, indices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool3dWithIndices(ctx, outWrapper, indicesWrapper, newInput, output_size);
}

inline diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    diopiConstTensorHandle_t newGrad_output,newInput,newIndices;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, indices, &newIndices, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiAdaptiveMaxPool3dBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newIndices);
}

inline diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {



    return ::diopiMaskedSelect(ctx, out, input, mask);
}

inline diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    diopiConstTensorHandle_t newGrad_output,newInput,newMask;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mask, &newMask, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaskedSelectBackward(ctx, grad_inputWrapper, newGrad_output, newInput, newMask);
}

inline diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMaximum(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMinimum(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    diopiConstTensorHandle_t newInput,newMat2;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mat2, &newMat2, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMm(ctx, outWrapper, newInput, newMat2);
}

inline diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIndexFillScalar(ctx, outWrapper, newInput, dim, newIndex, value);
}

inline diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    diopiConstTensorHandle_t newInput,newIndex,newValue;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, value, &newValue, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIndexFill(ctx, outWrapper, newInput, dim, newIndex, newValue);
}

inline diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    diopiConstTensorHandle_t newInput,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});

    return ::diopiIndexFillInpScalar(ctx, newInput, dim, newIndex, value);
}

inline diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    diopiConstTensorHandle_t newInput,newIndex,newValue;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, value, &newValue, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});

    return ::diopiIndexFillInp(ctx, newInput, dim, newIndex, newValue);
}

inline diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiExpand(ctx, outWrapper, newInput);
}

inline diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {


    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLinspace(ctx, outWrapper, start, end, steps);
}

inline diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPermute(ctx, outWrapper, newInput, dims);
}

inline diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode, double* value) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiPad(ctx, outWrapper, newInput, pad, mode, value);
}

inline diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRoll(ctx, outWrapper, newInput, shifts, dims);
}

inline diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiFlip(ctx, outWrapper, newInput, dims);
}

inline diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNorm(ctx, outWrapper, newInput, p, dim);
}

inline diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_meanWrapper = DiopiTensorWrapper<>(ctx, save_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_invstdWrapper = DiopiTensorWrapper<>(ctx, save_invstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGroupNorm(ctx, outWrapper, save_meanWrapper, save_invstdWrapper, newInput, newWeight, newBias, num_groups, eps);
}

inline diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, int64_t num_groups) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight,newMean,newRstd;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mean, &newMean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, rstd, &newRstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGroupNormBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, newMean, newRstd, num_groups);
}

inline diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted, bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {



    return ::diopiUnique(ctx, out, input, dim, sorted, return_counts, indices, counts);
}

inline diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiProd(ctx, outWrapper, newInput, dim);
}

inline diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    diopiConstTensorHandle_t newLog_probs,newTargets,newInput_lengths,newTarget_lengths;
    castImpl<diopiConstTensorHandle_t>(ctx, log_probs, &newLog_probs, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, targets, &newTargets, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input_lengths, &newInput_lengths, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target_lengths, &newTarget_lengths, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto neg_log_likelihoodWrapper = DiopiTensorWrapper<>(ctx, neg_log_likelihood, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto log_alphaWrapper = DiopiTensorWrapper<>(ctx, log_alpha, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCTCLoss(ctx, outWrapper, neg_log_likelihoodWrapper, log_alphaWrapper, newLog_probs, newTargets, newInput_lengths, newTarget_lengths, blank, reduction, zero_infinity);
}

inline diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    diopiConstTensorHandle_t newGrad_output,newLog_probs,newTargets,newInput_lengths,newTarget_lengths,newNeg_log_likelihood,newLog_alpha;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, log_probs, &newLog_probs, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, targets, &newTargets, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input_lengths, &newInput_lengths, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, target_lengths, &newTarget_lengths, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, neg_log_likelihood, &newNeg_log_likelihood, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, log_alpha, &newLog_alpha, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCTCLossBackward(ctx, grad_inputWrapper, newGrad_output, newLog_probs, newTargets, newInput_lengths, newTarget_lengths, newNeg_log_likelihood, newLog_alpha, blank, reduction, zero_infinity);
}

inline diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newInput,newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRemainderTensor(ctx, outWrapper, newInput, newOther);
}

inline diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRemainderScalar(ctx, outWrapper, newInput, other);
}

inline diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    diopiConstTensorHandle_t newOther;
    castImpl<diopiConstTensorHandle_t>(ctx, other, &newOther, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRemainder(ctx, outWrapper, input, newOther);
}

inline diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiConstTensorHandle_t newInput,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGather(ctx, outWrapper, newInput, dim, newIndex);
}

inline diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    diopiConstTensorHandle_t newGrad_output,newInput,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiGatherBackward(ctx, grad_inputWrapper, newGrad_output, newInput, dim, newIndex);
}

inline diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    diopiConstTensorHandle_t newSrc,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, src, &newSrc, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiScatterInp(ctx, inputWrapper, dim, newSrc, newIndex, reduce);
}

inline diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    diopiConstTensorHandle_t newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiScatterInpScalar(ctx, inputWrapper, dim, value, newIndex, reduce);
}

inline diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    diopiConstTensorHandle_t newInput,newSrc,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, src, &newSrc, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiScatter(ctx, outWrapper, newInput, dim, newSrc, newIndex, reduce);
}

inline diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    diopiConstTensorHandle_t newInput,newIndex;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, index, &newIndex, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiScatterScalar(ctx, outWrapper, newInput, dim, value, newIndex, reduce);
}

inline diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate) {
    std::vector<diopiConstTensorHandle_t> newIndices(indices_counts, diopiConstTensorHandle_t());
    for (int i = 0; i < indices_counts; ++i) {
        castImpl<diopiConstTensorHandle_t>(ctx, indices[i], &newIndices[i], std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    }
    diopiConstTensorHandle_t newValues;
    castImpl<diopiConstTensorHandle_t>(ctx, values, &newValues, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIndexPutInp(ctx, inputWrapper, newValues, newIndices.data(), indices_counts, accumulate);
}

inline diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate) {
    std::vector<diopiConstTensorHandle_t> newIndices(indices_counts, diopiConstTensorHandle_t());
    for (int i = 0; i < indices_counts; ++i) {
        castImpl<diopiConstTensorHandle_t>(ctx, indices[i], &newIndices[i], std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    }
    diopiConstTensorHandle_t newInput,newValues;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, values, &newValues, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIndexPut(ctx, outWrapper, newInput, newValues, newIndices.data(), indices_counts, accumulate);
}

inline diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {


    auto inoutWrapper = DiopiTensorWrapper<>(ctx, inout, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRandomInp(ctx, inoutWrapper, from, to, idx);
}

inline diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {


    auto inoutWrapper = DiopiTensorWrapper<>(ctx, inout, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUniformInp(ctx, inoutWrapper, from, to, idx);
}

inline diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBernoulli(ctx, outWrapper, newInput, idx);
}

inline diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {


    auto inoutWrapper = DiopiTensorWrapper<>(ctx, inout, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBernoulliInp(ctx, inoutWrapper, idx);
}

inline diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {


    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiBernoulliScalar(ctx, outWrapper, p, idx);
}

inline diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {


    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiArange(ctx, outWrapper, start, end, step);
}

inline diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {


    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRandperm(ctx, outWrapper, n, idx);
}

inline diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std) {


    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNormal(ctx, outWrapper, mean, std);
}

inline diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std) {
    diopiConstTensorHandle_t newMean;
    castImpl<diopiConstTensorHandle_t>(ctx, mean, &newMean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNormalTensorScalar(ctx, outWrapper, newMean, std);
}

inline diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std) {
    diopiConstTensorHandle_t newStd;
    castImpl<diopiConstTensorHandle_t>(ctx, std, &newStd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNormalScalarTensor(ctx, outWrapper, mean, newStd);
}

inline diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std) {
    diopiConstTensorHandle_t newMean,newStd;
    castImpl<diopiConstTensorHandle_t>(ctx, mean, &newMean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, std, &newStd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNormalTensor(ctx, outWrapper, newMean, newStd);
}

inline diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std) {


    auto inoutWrapper = DiopiTensorWrapper<>(ctx, inout, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiNormalInp(ctx, inoutWrapper, mean, std);
}

inline diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum) {



    return ::diopiMeshGrid(ctx, outs, inputs, inputsNum);
}

inline diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiMultinomial(ctx, outWrapper, newInput, num_samples, replacement);
}

inline diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalized_shape, double eps) {
    diopiConstTensorHandle_t newInput,newWeight,newBias;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_meanWrapper = DiopiTensorWrapper<>(ctx, save_mean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto save_invstdWrapper = DiopiTensorWrapper<>(ctx, save_invstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLayerNorm(ctx, outWrapper, save_meanWrapper, save_invstdWrapper, newInput, newWeight, newBias, normalized_shape, eps);
}

inline diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    diopiConstTensorHandle_t newGrad_output,newInput,newWeight,newBias,newMean,newRstd;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, weight, &newWeight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, bias, &newBias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, mean, &newMean, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    castImpl<diopiConstTensorHandle_t>(ctx, rstd, &newRstd, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_weightWrapper = DiopiTensorWrapper<>(ctx, grad_weight, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_biasWrapper = DiopiTensorWrapper<>(ctx, grad_bias, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiLayerNormBackward(ctx, grad_inputWrapper, grad_weightWrapper, grad_biasWrapper, newGrad_output, newInput, newWeight, newBias, newMean, newRstd, normalized_shape);
}

inline diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {
    diopiConstTensorHandle_t newSrc;
    castImpl<diopiConstTensorHandle_t>(ctx, src, &newSrc, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCopyInp(ctx, newSrc, inputWrapper);
}

inline diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUpsampleNearest(ctx, outWrapper, newInput, size);
}

inline diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size) {
    diopiConstTensorHandle_t newGrad_output;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUpsampleNearestBackward(ctx, grad_inputWrapper, newGrad_output, out_size, in_size);
}

inline diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool align_corners, const char* mode) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUpsampleLinear(ctx, outWrapper, newInput, size, align_corners, mode);
}

inline diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode) {
    diopiConstTensorHandle_t newGrad_output;
    castImpl<diopiConstTensorHandle_t>(ctx, grad_output, &newGrad_output, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto grad_inputWrapper = DiopiTensorWrapper<>(ctx, grad_input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiUpsampleLinearBackward(ctx, grad_inputWrapper, newGrad_output, out_size, in_size, align_corners, mode);
}

inline diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiErfinv(ctx, outWrapper, newInput);
}

inline diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {


    auto inputWrapper = DiopiTensorWrapper<>(ctx, input, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiErfinvInp(ctx, inputWrapper);
}

inline diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiIm2Col(ctx, outWrapper, newInput, kernel_size, dilation, padding, stride);
}

inline diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size, diopiSize_t kernel_size, diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCol2Im(ctx, outWrapper, newInput, output_size, kernel_size, dilation, padding, stride);
}

inline diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiRepeat(ctx, outWrapper, newInput, repeats_size);
}

inline diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    diopiConstTensorHandle_t newInput;
    castImpl<diopiConstTensorHandle_t>(ctx, input, &newInput, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    auto outWrapper = DiopiTensorWrapper<>(ctx, out, std::vector<diopiMemoryFormat_t>{diopiMemoryFormat_t::ChannelsLast});
    return ::diopiCastDtype(ctx, outWrapper, newInput);
}

}
# endif // DIOPI_ADAPTOR_HPP