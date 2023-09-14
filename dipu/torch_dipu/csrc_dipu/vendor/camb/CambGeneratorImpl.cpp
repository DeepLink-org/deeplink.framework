// Copyright (c) 2023, DeepLink.
#include <ATen/Utils.h>
#include <ATen/Functions.h>

#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>
#include <csrc_dipu/runtime/core/DIPUGuard.h>

#include <cnnl.h>

#include <iostream>

namespace dipu {

static size_t mlu_state_size = []() {
  size_t size = 0;
  DIPU_CALLCNNL(cnnlRandGetMTGP32StateSize(nullptr, &size));
  return size;
}();

static deviceHandle_t getDeviceHandler(c10::DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = devapis::current_device();
  }
  deviceHandle_t handle;
  DIPU_CALLCNNL(cnnlCreate(&handle));
  auto stream = getCurrentDIPUStream(device_index);
  DIPU_CALLCNNL(cnnlSetQueue(handle, stream.rawstream()));
  return handle;
}
  
// Discriminate floating device type.
static bool is_floating_device = true;

class MLUGeneratorImpl : public dipu::DIPUGeneratorImpl {
public:
  MLUGeneratorImpl(at::DeviceIndex device_index): dipu::DIPUGeneratorImpl(device_index) {
  }

  /**
  * set state
  *
  * See Note [Acquire lock when using random generators]
  */
  void set_state(const c10::TensorImpl& state) override {
    auto state_size = state.numel();
    TORCH_CHECK(state_size == mlu_state_size, "RNG state is wrong size");

    at::Tensor state_tmp(state.shallow_copy_and_detach(state.version_counter(), true));
    state_ = state_tmp.to(device_);
    state_need_reset_ = false;
  }

  /**
   * update_state
   *
   * See Note [Acquire lock when using random generators]
    */
  void update_state() const override {
    // update the state tensor.
    TORCH_CHECK(is_floating_device, "is_floating_device must be true");
    if (!state_need_reset_) {
      return;
    }

    if (!state_.defined()) {
      auto options = at::TensorOptions().device(device_).dtype(at::kByte);
      state_ = at::empty(mlu_state_size, options);
    }
    auto state_ptr = state_.tensor_data().data_ptr();
    dipu::DIPUGuard guard(state_.device());
    auto handle = getDeviceHandler(state_.device().index());
    DIPU_CALLCNNL(cnnlRandMakeMTGP32KernelState(handle, state_ptr, nullptr, nullptr, seed_));
    state_need_reset_ = false;
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<MLUGeneratorImpl>(device_index);
}

}  // namespace torch_dipu