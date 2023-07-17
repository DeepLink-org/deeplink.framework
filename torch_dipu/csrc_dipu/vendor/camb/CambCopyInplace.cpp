// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/common.h>
#include <c10/util/Exception.h>
#include <csrc_dipu/runtime/ops/DIPUCopyInplace.h>

namespace dipu {

// TODO(caikun): move to diopi autogen
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

  // TODO(caikun): remove syncStream when cache allocator is ready
  if (non_blocking) {
    dipu::devapis::syncStream(stream.rawstream());
  }
  return self;
}

class CambCopyInplace : public DIPUCopyInplace {
public:
  CambCopyInplace() = default;
  ~CambCopyInplace() = default;

  void copy_between_devices(at::Tensor& self, const at::Tensor& src, at::TensorIterator& iter, bool non_blocking) override {
    std::cout << "enter into CambCopyInplace::copy_between_devices" << std::endl;
    at::Tensor src_expand = src.expand_as(self);
    copy_(self, src_expand, non_blocking);
  }
};

static CambCopyInplace camb_copy_inplace;
static int32_t camb_init = [&]() {
    setDipuCopyInplace(&camb_copy_inplace);
    return 1;
}();

}  // namespace dipu