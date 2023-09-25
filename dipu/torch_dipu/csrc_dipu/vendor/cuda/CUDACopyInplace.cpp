// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/common.h>
#include <c10/util/Exception.h>
#include <csrc_dipu/runtime/core/DIPUCopyInplace.h>

namespace dipu {

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  if (self.numel() == 0) {
    return self;
  }

  dipu::DIPUStream stream = getCurrentDIPUStream();
  ::diopiContext context(stream.rawstream());
  auto ctx = &context;

  ::diopiConstTensorHandle_t srcDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(src);
  ::diopiTensorHandle_t selfDiopiTensorHandle = dipu::diopi_helper::toDiopiTensorHandle(self);
  ::diopiError_t ret = ::diopiCopyInp(ctx, srcDiopiTensorHandle, selfDiopiTensorHandle);
  TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, R"(::diopiCopyInp(ctx, src, dst);)", " error, error code is ", ret, "error message is ", diopiGetLastErrorString());

  if (!non_blocking) {
    dipu::devapis::syncStream(stream.rawstream());
  }
  return self;
}

class CUDACopyInplace : public DIPUCopyInplace {
public:
  CUDACopyInplace() = default;
  ~CUDACopyInplace() = default;

  at::Tensor& run(at::Tensor& self, const at::Tensor& src, bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }

  at::Tensor& copy_between_devices(at::TensorIterator& iter, at::Tensor& self, const at::Tensor& src, bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }

  at::Tensor& copy_contiguous(at::TensorIterator& iter, at::Tensor& self, const at::Tensor& src, bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }

  at::Tensor& copy_uncontiguous(at::TensorIterator& iter, at::Tensor& self, const at::Tensor& src, bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }
};

static CUDACopyInplace cuda_copy_inplace;
static int32_t cuda_init = []() {
  setDipuCopyInplace(&cuda_copy_inplace);
  return 1;
}();

}  // namespace dipu
