// Copyright (c) 2023, DeepLink.
#include <stdio.h>

#include "./diopirt_impl.h"

namespace diopihelper = dipu::diopi_helper;

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
    *size = diopiSize_t(ptr->sizes().data(), ptr->dim());
    return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGetTensorStride(diopiConstTensorHandle_t pth, diopiSize_t* stride) {
    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(pth);
    *stride = diopiSize_t(ptr->strides().data(), ptr->dim());
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
    diopiSize_t size(&num_bytes, 1);
    return diopiRequireTensor(ctx, tensor, &size, nullptr, diopi_dtype_int8, device);
}

DIOPI_RT_API diopiError_t diopiGeneratorInitState(diopiConstGeneratorHandle_t th) {
//   std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
//   const at::Generator* generator = reinterpret_cast<const at::Generator*>(th);
//   const dipu::DIPUGeneratorImpl* impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);
//   impl->init_state();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorUpdateState(diopiConstGeneratorHandle_t th) {
//   std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
//   const at::Generator* generator = reinterpret_cast<const at::Generator*>(th);
//   dipu::DIPUGeneratorImpl* impl = at::check_generator<dipu::DIPUGeneratorImpl>(*generator);
//   impl->update_state();
  return diopiSuccess;
}

DIOPI_RT_API diopiError_t diopiGeneratorGetState(diopiConstGeneratorHandle_t th, void **data) {
  std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  const at::Generator* generator = reinterpret_cast<const at::Generator*>(th);
  // TODO(caikun): add lock
  *data = generator->get_state().data_ptr();
  return diopiSuccess;
}

}  // extern "C"