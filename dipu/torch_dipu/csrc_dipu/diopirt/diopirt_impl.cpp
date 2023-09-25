// Copyright (c) 2023, DeepLink.
#include <stdio.h>
#include <mutex>

#include "./diopirt_impl.h"
#include "csrc_dipu/profiler/profiler.h"

namespace diopihelper = dipu::diopi_helper;
using dipu::profile::RecordBlockCreator;

extern "C" {

static char diopiVersion[256] = {0};

DIOPI_RT_API const char* diopiGetVersion() {
    static bool inited = false;
    if (!inited) {
        inited = true;
        snprintf(diopiVersion, sizeof(diopiVersion), "DIOPI Version: %d.%d.%d", DIOPI_VER_MAJOR, DIOPI_VER_MINOR, DIOPI_VER_PATCH);
    }
    return diopiVersion;
}

DIOPI_RT_API diopiError_t diopiGetTensorData(diopiTensorHandle_t pth, void** pptr) {
    *pptr = (reinterpret_cast<at::Tensor*>(pth))->data_ptr();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDataConst(diopiConstTensorHandle_t pth, const void** pptr) {
    *pptr = (reinterpret_cast<const at::Tensor*>(pth))->data_ptr();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorShape(diopiConstTensorHandle_t pth, diopiSize_t* size) {
    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(pth);
    *size = diopiSize_t{ptr->sizes().data(), static_cast<int64_t>(ptr->dim())};
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t pth, diopiSize_t* stride) {
    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(pth);
    *stride = diopiSize_t{ptr->strides().data(), static_cast<int64_t>(ptr->dim())};
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDtype(diopiConstTensorHandle_t pth, diopiDtype_t* dtype) {
    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(pth);
    *dtype = diopihelper::toDiopiDtype(ptr->scalar_type());
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorDevice(diopiConstTensorHandle_t pth, diopiDevice_t* device) {
    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(pth);
    *device = (ptr->is_cpu() ? diopi_host : diopi_device);
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorNumel(diopiConstTensorHandle_t pth, int64_t* numel) {
    if (pth == nullptr) {
        *numel = 0;
        return diopiSuccess;
    }

    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(pth);
    *numel = ptr->numel();
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorElemSize(diopiConstTensorHandle_t pth, int64_t* elemsize) {
    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(pth);
    diopiDtype_t dtype;
    auto ret = diopiGetTensorDtype(pth, &dtype);
    if (ret != diopiSuccess) return ret;

    *elemsize = diopihelper::getElemSize(dtype);
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetStream(diopiContextHandle_t ctx, diopiStreamHandle_t* stream) {
    *stream = ctx->stream;
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireTensor(
    diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
    const diopiSize_t* size, const diopiSize_t* stride,
    const diopiDtype_t dtype, const diopiDevice_t device) {
    // TORCH_CHECK(tensor != nullptr && *tensor == nullptr, "invalid parameter tensor");
    at::IntArrayRef at_dims(size->data, size->len);
    caffe2::TypeMeta at_type = diopihelper::toATenType(dtype);
    c10::DeviceType at_device = diopihelper::toATenDevice(device);
    auto options = at::TensorOptions(at_device).dtype(at_type);
    at::Tensor t;
    if (stride) {
        at::IntArrayRef at_stride(stride->data, stride->len);
        t = at::empty_strided(at_dims, at_stride, options);
    } else {
        t = at::empty(at_dims, options);
    }

    ctx->arrays.emplace_back(std::move(t));
    *tensor = reinterpret_cast<diopiTensorHandle_t>(&(ctx->arrays.back()));
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRequireBuffer(
    diopiContextHandle_t ctx, diopiTensorHandle_t* tensor,
    int64_t num_bytes, diopiDevice_t device) {
    diopiSize_t size{&num_bytes, 1};
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, device);
}

DIOPI_RT_API diopiError_t diopiGeneratorGetState(diopiContextHandle_t ctx, diopiConstGeneratorHandle_t th, diopiTensorHandle_t *state) {
  const at::Generator* generator = reinterpret_cast<const at::Generator*>(th);
  dipu::DIPUGeneratorImpl* gen_impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);

  at::Tensor tensor;
  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    tensor = at::Tensor::wrap_tensor_impl(gen_impl->get_state());
  }

  ctx->arrays.emplace_back(std::move(tensor));
  *state = reinterpret_cast<diopiTensorHandle_t>(&(ctx->arrays.back()));
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorSetState(diopiGeneratorHandle_t th, diopiConstTensorHandle_t new_state) {
  at::Generator* generator = reinterpret_cast<at::Generator*>(th);
  dipu::DIPUGeneratorImpl* gen_impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);
  const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(new_state);

  {
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    gen_impl->set_state(*(ptr->unsafeGetTensorImpl()));
  }

  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRecordStart(const char* record_name, void** record) {
    *record = new RecordBlockCreator(record_name);
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiRecordEnd(void** record) {
    TORCH_CHECK(record != nullptr, "invalid parameter record_function");
    auto dipu_record_block = static_cast<RecordBlockCreator*>(*record);
    dipu_record_block->end();
    delete dipu_record_block;
    *record = nullptr;
    return diopiSuccess;
}

}  // extern "C"
