// Copyright (c) 2023, DeepLink.
#include <ATen/Utils.h>
#include <ATen/Functions.h>

#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>
#include <csrc_dipu/runtime/core/DIPUGuard.h>

#include <cnnl.h>

#include <iostream>

namespace dipu {

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
protected:
  mutable std::once_flag init_state_flag;
public:
  MLUGeneratorImpl(at::DeviceIndex device_index): dipu::DIPUGeneratorImpl(device_index) {
  }
  /**
   * get_init_state_flag
   *
   * See Note [Acquire lock when using random generators]
   */
  void init_state() const override {
    std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
    // resize and set the state tensor.
    TORCH_CHECK(is_floating_device, "MLUGeneratorImpl only support on floating device");
    std::call_once(init_state_flag, [&] {
      size_t state_size = 0;
      DIPU_CALLCNNL(cnnlRandGetMTGP32StateSize(nullptr, &state_size));
      auto options = at::TensorOptions().device(device_).dtype(at::kByte);
      state_ = at::empty(state_size, options);
      std::cout << "init state, state size=" << state_size << std::endl;
    });
  }

  /**
  * set state
  *
  * See Note [Acquire lock when using random generators]
  */
  void set_state(const c10::TensorImpl& state) override {
    std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
    // at::detail::check_rng_state(state);
    // 5056 is numel() of a cpu state tensor, 816 is gpu's and 1180672 is mlu's,
    // hardcoding the number just like the original impl.
    const int cpu_numel = 5056;
    const int gpu_numel = 816;
    const int mlu_numel = 1180672;
    at::Tensor state_tmp(state.shallow_copy_and_detach(state.version_counter(), true));
    if (state_tmp.numel() == cpu_numel || state_tmp.numel() == gpu_numel) {
        return;
    } else if (state_tmp.numel() == mlu_numel) {
        init_state();
        state_ = state_tmp.to(state_.device());
        state_need_reset_ = false;
    } else {
        TORCH_CHECK(false, "RNG state is wrong size.");
    }
  }

  /**
   * update_state
   *
   * See Note [Acquire lock when using random generators]
    */
  void update_state() const override {
    std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
    // update the state tensor.
    if (is_floating_device && state_need_reset_) {
      auto state_ptr = state_.tensor_data().data_ptr();
      TORCH_CHECK(state_ptr, "the state point is nullptr, "
                            "please init state before calling its point");
      dipu::DIPUGuard guard(state_.device());
      auto handle = getDeviceHandler(state_.device().index());
      DIPU_CALLCNNL(cnnlRandMakeMTGP32KernelState(handle, state_ptr, nullptr, nullptr, seed_));
      state_need_reset_ = false;
    }
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  std::cout << "enter into " << __FILE__ << ":" << __FUNCTION__ << std::endl;
  return at::make_generator<MLUGeneratorImpl>(device_index);
}

}  // namespace torch_dipu