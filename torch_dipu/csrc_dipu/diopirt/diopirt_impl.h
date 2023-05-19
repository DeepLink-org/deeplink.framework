// Copyright (c) 2023, DeepLink.
#pragma once

#include <list>
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>

#include <diopi/diopirt.h>
#include <diopi/functions.h>

#include <csrc_dipu/runtime/rthelper.h>

using deviceStream_t = dipu::deviceStream_t;

extern "C" {
struct diopiContext {
    // TODO(caikun): use dipu stream? can only use device stream?
    deviceStream_t stream;
    // 1. use arrays to hold tensor that avoid tensor deleting when leaving scope
    // 2. The address of each array must be fixed, so use list instead of vector
    std::list<at::Tensor> arrays;

    explicit diopiContext(const deviceStream_t& s) : stream(s) {}
};

}  // extern "C"

namespace dipu {

namespace diopi_helper {

::diopiTensorHandle_t toDiopiTensorHandle(at::Tensor& tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor& tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor* tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const c10::optional<at::Tensor>& tensor);

::diopiScalar_t toDiopiScalar(const at::Scalar& scalar);
::diopiDtype_t toDiopiDtype(c10::ScalarType type);

caffe2::TypeMeta toATenType(::diopiDtype_t dt);
int64_t getElemSize(::diopiDtype_t dt);

c10::DeviceType toATenDevice(::diopiDevice_t device);

::diopiSize_t toDiopiSize(const at::OptionalIntArrayRef& dim);

::diopiRoundMode_t toDiopiRoundMode(const std::string& rounding_mode);

}  // namespace diopi_helper

}  // namespace dipu