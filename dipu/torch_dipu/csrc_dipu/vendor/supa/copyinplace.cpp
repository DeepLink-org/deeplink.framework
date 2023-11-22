// Copyright (c) 2023, DeepLink.

#include <csrc_dipu/common.h>
#include <csrc_dipu/diopirt/diopirt_impl.h>
#include <csrc_dipu/runtime/core/DIPUCopyInplace.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>

namespace dipu {
namespace supa {

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  if (self.numel() == 0) {
    return self;
  }
  dipu::DIPUStream stream = getCurrentDIPUStream();
  ::diopiContext context(stream.rawstream());
  auto ctx = &context;
  ::diopiConstTensorHandle_t srcDiopiTensorHandle =
      dipu::diopi_helper::toDiopiTensorHandle(src);
  ::diopiTensorHandle_t selfDiopiTensorHandle =
      dipu::diopi_helper::toDiopiTensorHandle(self);
  ::diopiError_t ret =
      ::diopiCopyInp(ctx, srcDiopiTensorHandle, selfDiopiTensorHandle);
  TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__,
              R"(::diopiCopyInp(ctx, src, dst);)", " error, error code is ",
              ret, "error message is ", diopiGetLastErrorString());
  // TODO(caikun): remove syncStream when cache allocator is ready
  if (non_blocking) {
    dipu::devapis::syncStream(stream.rawstream());
  }
  return self;
}

class SUPACopyInplace : public DIPUCopyInplace {
 public:
  SUPACopyInplace() = default;
  ~SUPACopyInplace() = default;

  at::Tensor& run(at::Tensor& self, const at::Tensor& src,
                  bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }

  at::Tensor& copy_between_devices(at::TensorIterator& iter, at::Tensor& self,
                                   const at::Tensor& src,
                                   bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }

  at::Tensor& copy_contiguous(at::TensorIterator& iter, at::Tensor& self,
                              const at::Tensor& src,
                              bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }

  at::Tensor& copy_uncontiguous(at::TensorIterator& iter, at::Tensor& self,
                                const at::Tensor& src,
                                bool non_blocking) override {
    return copy_(self, src, non_blocking);
  }
};

static SUPACopyInplace copy_inplace;
static int32_t suap_copy_inplace_init = []() {
  setDipuCopyInplace(&copy_inplace);
  return 1;
}();

}  // namespace supa
}  // namespace dipu
