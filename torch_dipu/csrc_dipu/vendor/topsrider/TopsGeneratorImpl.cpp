// Copyright (c) 2023, DeepLink.
#include <ATen/Utils.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

namespace dipu {
// just an example
class TopsGeneratorImpl : public dipu::DIPUGeneratorImpl {
public:
  TopsGeneratorImpl(at::DeviceIndex device_index): dipu::DIPUGeneratorImpl(device_index) {
  }

  void init_state() const override {
  }

  void set_state(const c10::TensorImpl& state) override {
  }

  void update_state() const override {
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<TopsGeneratorImpl>(device_index);
}

}  // namespace torch_dipu