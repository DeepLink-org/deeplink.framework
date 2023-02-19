#include "csrc_dipu/diopirt/diopi.h"

#include <stdio.h>

namespace dipu {

namespace diopi {

::diopiTensorHandle_t toDiopiTensorHandle(at::Tensor& tensor) {
    return reinterpret_cast<::diopiTensorHandle_t>(&tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor& tensor) {
    return reinterpret_cast<::diopiConstTensorHandle_t>(&tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor* tensor) {
    return tensor == nullptr ? nullptr : reinterpret_cast<::diopiConstTensorHandle_t>(tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const c10::optional<at::Tensor>& tensor) {
    if (!tensor.has_value()) return nullptr;
    return reinterpret_cast<::diopiConstTensorHandle_t>(&(tensor.value()));
}

::diopiScalar_t toDiopiScalar(const at::Scalar& scalar) {
    ::diopiScalar_t result;
    switch (scalar.type()) {
    case c10::ScalarType::Bool: {
        result.stype = ::diopiDtype_t::diopi_dtype_int64;
        result.ival = static_cast<int64_t>(scalar.toBool());
        return result;
    }
    case c10::ScalarType::Long: {
        result.stype = ::diopiDtype_t::diopi_dtype_int64;
        result.ival = static_cast<int64_t>(scalar.toLong());
        return result;
    }
    case c10::ScalarType::Double: {
        result.stype = ::diopiDtype_t::diopi_dtype_float64;
        result.fval = scalar.toDouble();
        return result;
    }
    default: {
        TORCH_CHECK(false, "invalid scalar type, type is ", scalar.type());
        break;
    }
    }
}

::diopiDtype_t toDiopiDtype(c10::ScalarType type) {
    switch (type) {
    case at::ScalarType::Char:
        return diopi_dtype_bool;
    case at::ScalarType::Byte:
        return diopi_dtype_uint8;
    case at::ScalarType::Short:
        return diopi_dtype_int16;
    case at::ScalarType::Int:
        return diopi_dtype_int32;
    case at::ScalarType::Long:
        return diopi_dtype_int64;
    case at::ScalarType::Half:
        return diopi_dtype_float16;
    case at::ScalarType::Float:
        return diopi_dtype_float32;
    case at::ScalarType::Double:
        return diopi_dtype_float64;
    default:
        TORCH_CHECK(false, "invalid scalar type, type is ", type);
    }
}

caffe2::TypeMeta toATenType(::diopiDtype_t dt) {
    switch (dt) {
    case diopi_dtype_bool:
        return caffe2::TypeMeta::Make<bool>();
    case diopi_dtype_uint8:
        return caffe2::TypeMeta::Make<uint8_t>();
    case diopi_dtype_int8:
        return caffe2::TypeMeta::Make<int8_t>();
    case diopi_dtype_int16:
        return caffe2::TypeMeta::Make<int16_t>();
    case diopi_dtype_uint16:
        return caffe2::TypeMeta::Make<uint16_t>();
    case diopi_dtype_int32:
    case  diopi_dtype_uint32:
        return caffe2::TypeMeta::Make<int32_t>();
    case diopi_dtype_int64:
    case diopi_dtype_uint64:
        return caffe2::TypeMeta::Make<int64_t>();
        return caffe2::TypeMeta::Make<uint64_t>();
    case diopi_dtype_float32:
        return caffe2::TypeMeta::Make<float>();
    case diopi_dtype_float64:
        return caffe2::TypeMeta::Make<double>();
    case diopi_dtype_float16:
        return caffe2::TypeMeta::Make<at::Half>();
    case diopi_dtype_bfloat16:
        return caffe2::TypeMeta::Make<at::BFloat16>();
    default:
        TORCH_CHECK(false, "invalid diopi type, diopi type is ", dt);
    }
}

int64_t getElemSize(::diopiDtype_t dt) {
    switch (dt) {
    case diopi_dtype_int32:
    case diopi_dtype_uint32:
    case diopi_dtype_float32:
    case diopi_dtype_tfloat32:
        return 4;
    case diopi_dtype_int64:
    case diopi_dtype_uint64:
    case diopi_dtype_float64:
        return 8;
    case diopi_dtype_int16:
    case diopi_dtype_uint16:
    case diopi_dtype_float16:
    case diopi_dtype_bfloat16:
        return 2;
    case diopi_dtype_int8:
    case diopi_dtype_uint8:
    case diopi_dtype_bool:
        return 1;
    default:
        TORCH_CHECK(false, "invalid diopi type, diopi type is ", dt);
    }
}

c10::DeviceType toATenDevice(::diopiDevice_t device) {
    switch (device) {
    case diopi_host:
        return c10::DeviceType::CPU;
    case diopi_device:
        return c10::DeviceType::CUDA;
    default:
        TORCH_CHECK(false, "invalid diopi device, diopi device is ", device);
    }
}

at::Tensor fromPreAllocated(
    void* data, at::IntArrayRef sizes,
    at::IntArrayRef strides, const std::function<void(void*)>& deleter,
    at::Allocator* allocator, const at::TensorOptions& options) {
    auto device = at::globalContext().getDeviceFromPtr(data, options.device().type());
    if (options.device().has_index()) {
        assert(options.device() == device);
    }

    auto storage = at::Storage(
        at::Storage::use_byte_size_t(),
        at::detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize()),
        c10::InefficientStdFunctionContext::makeDataPtr(data, deleter, device),
        allocator, false);
    return at::empty({0}, options).set_(storage, 0, sizes, strides);
}

}  // namespace diopi

}  // namespace dipu