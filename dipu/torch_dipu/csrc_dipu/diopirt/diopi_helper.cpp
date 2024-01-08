// Copyright (c) 2023, DeepLink.
#include <cstdio>

#include "./diopirt_impl.h"

namespace dipu {

namespace diopi_helper {

at::Tensor* fromDiopiTensorHandle(::diopiTensorHandle_t tensor) {
  return reinterpret_cast<at::Tensor*>(tensor);
}

::diopiTensorHandle_t toDiopiTensorHandle(at::Tensor& tensor) {
  return tensor.defined() ? reinterpret_cast<::diopiTensorHandle_t>(&tensor)
                          : nullptr;
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor& tensor) {
  return tensor.defined()
             ? reinterpret_cast<::diopiConstTensorHandle_t>(&tensor)
             : nullptr;
}

::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor* tensor) {
  return tensor == nullptr ? nullptr : toDiopiTensorHandle(*tensor);
}

::diopiConstTensorHandle_t toDiopiTensorHandle(
    const c10::optional<at::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return nullptr;
  }
  return toDiopiTensorHandle(tensor.value());
}

::diopiGeneratorHandle_t toDiopiGeneratorHandle(at::Generator& generator) {
  return generator.defined()
             ? reinterpret_cast<::diopiGeneratorHandle_t>(&generator)
             : nullptr;
}

::diopiGeneratorHandle_t toDiopiGeneratorHandle(
    c10::optional<at::Generator>& generator) {
  if (!generator.has_value()) {
    return nullptr;
  }
  return toDiopiGeneratorHandle(generator.value());
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

::diopiScalar_t toDiopiScalar(const at::Scalar& scalar,
                              const c10::ScalarType& type) {
  ::diopiScalar_t result;
  TORCH_CHECK(c10::canCast(scalar.type(), type));
  if (type == c10::ScalarType::Bool) {
    result.stype = ::diopiDtype_t::diopi_dtype_int64;
    result.ival = static_cast<int64_t>(scalar.toBool());
    return result;
  }
  if (c10::isFloatingType(type)) {
    result.stype = ::diopiDtype_t::diopi_dtype_float64;
    result.fval = scalar.toDouble();
    return result;
  }
  if (c10::isIntegralType(type, false)) {
    result.stype = ::diopiDtype_t::diopi_dtype_int64;
    result.ival = static_cast<int64_t>(scalar.toLong());
    return result;
  }
  TORCH_CHECK(false, "invalid scalar type, type is ", scalar.type());
}

::diopiDtype_t toDiopiDtype(c10::ScalarType type) {
  switch (type) {
    case at::ScalarType::Bool:
      return diopi_dtype_bool;
    case at::ScalarType::Char:
      return diopi_dtype_int8;
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
    case at::ScalarType::BFloat16:
      return diopi_dtype_bfloat16;
    case at::ScalarType::Float:
      return diopi_dtype_float32;
    case at::ScalarType::Double:
      return diopi_dtype_float64;
    case at::ScalarType::ComplexFloat:
      return diopi_dtype_complex64;
    case at::ScalarType::ComplexDouble:
      return diopi_dtype_complex128;
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
    case diopi_dtype_uint32:
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
    case diopi_dtype_complex64:
      return caffe2::TypeMeta::Make<c10::complex<float>>();
    case diopi_dtype_complex128:
      return caffe2::TypeMeta::Make<c10::complex<double>>();
    default:
      TORCH_CHECK(false, "invalid diopi type, diopi type is ", dt);
  }
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
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
    case diopi_dtype_complex64:
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
    case diopi_dtype_complex128:
      return 16;
    default:
      TORCH_CHECK(false, "invalid diopi type, diopi type is ", dt);
  }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)

c10::DeviceType toATenDevice(::diopiDevice_t device) {
  switch (device) {
    case diopi_host:
      return c10::DeviceType::CPU;
    case diopi_device:
      return dipu::DIPU_DEVICE_TYPE;
    default:
      TORCH_CHECK(false, "invalid diopi device, diopi device is ", device);
  }
}

::diopiSize_t toDiopiSize(const at::OptionalIntArrayRef& dim) {
  ::diopiSize_t diopi_size{nullptr, 0};
  if (dim.has_value()) {
    diopi_size.data = dim.value().data();
    diopi_size.len = static_cast<int64_t>(dim.value().size());
  }
  return diopi_size;
}

::diopiSize_t toDiopiSize(at::IntArrayRef input) {
  ::diopiSize_t diopi_size{nullptr, 0};
  diopi_size.data = input.data();
  diopi_size.len = static_cast<int64_t>(input.size());
  return diopi_size;
}

::diopiRoundMode_t toDiopiRoundMode(const std::string& rounding_mode) {
  if (rounding_mode == "none" || rounding_mode == "None" ||
      rounding_mode.empty()) {
    return RoundModeNone;
  }
  if (rounding_mode == "floor") {
    return RoundModeFloor;
  }
  if (rounding_mode == "trunc") {
    return RoundModeTrunc;
  }
  TORCH_CHECK(false,
              "rounding_mode should be none, 'floor' or 'trunc', but got ",
              rounding_mode)
}

}  // namespace diopi_helper

}  // namespace dipu
