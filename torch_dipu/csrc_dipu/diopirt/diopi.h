#pragma once

#include <third_party/DIOPI/include/diopi/diopirt.h>
#include <third_party/DIOPI/include/diopi/functions.h>

#include <list>

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <c10/core/Scalar.h>
#include <cuda_runtime_api.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAStream.h>

extern "C" {
struct diopiContext {
    // TODO(caikun): use dipu stream? can only use device stream?
    cudaStream_t stream;
    // 1. use arrays to hold tensor that avoid tensor deleting when leaving scope 
    // 2. The address of each array must be fixed, so use list instead of vector
    std::list<at::Tensor> arrays;

    explicit diopiContext(const cudaStream_t& s) : stream(s) {}
};

}  // extern "C"

namespace dipu {

namespace diopi {

::diopiTensorHandle_t toDiopiTensorHandle(at::Tensor& tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor& tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor* tensor);
::diopiConstTensorHandle_t toDiopiTensorHandle(const c10::optional<at::Tensor>& tensor);

::diopiScalar_t toDiopiScalar(const at::Scalar& scalar);
::diopiDtype_t toDiopiDtype(c10::ScalarType type);

caffe2::TypeMeta toATenType(::diopiDtype_t dt);
int64_t getElemSize(::diopiDtype_t dt);

c10::DeviceType toATenDevice(::diopiDevice_t device);

at::Tensor fromPreAllocated(
    void* data, at::IntArrayRef sizes,
    at::IntArrayRef strides, const std::function<void(void*)>& deleter,
    at::Allocator* allocator, const at::TensorOptions& options);

}  // namespace diopi

}  // namespace dipu