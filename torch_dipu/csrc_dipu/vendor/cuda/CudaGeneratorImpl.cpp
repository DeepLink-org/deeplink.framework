// Copyright (c) 2023, DeepLink.
#include <ATen/Utils.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

#include <iostream>
namespace dipu {
// just an example
class CUDAGeneratorImpl : public dipu::DIPUGeneratorImpl {
public:
  CUDAGeneratorImpl(at::DeviceIndex device_index): dipu::DIPUGeneratorImpl(device_index) {
  }

  void init_state() const override {
    std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  }

  void set_state(const c10::TensorImpl& state) override {
    std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  }

  void update_state() const override {
    std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  return at::make_generator<CUDAGeneratorImpl>(device_index);
}

}  // namespace torch_dipu