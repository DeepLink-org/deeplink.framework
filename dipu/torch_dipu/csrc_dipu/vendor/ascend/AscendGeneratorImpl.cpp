// Copyright (c) 2023, DeepLink.
#include <ATen/Functions.h>
#include <ATen/Utils.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>
#include <csrc_dipu/runtime/core/DIPUGuard.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

namespace dipu {

static const size_t seed_size = sizeof(uint64_t);
static const size_t offset_size = sizeof(int64_t);
static const size_t total_size = seed_size + offset_size;

class NPUGeneratorImpl : public dipu::DIPUGeneratorImpl {
 protected:
  mutable std::once_flag init_state_flag;

 public:
  NPUGeneratorImpl(at::DeviceIndex device_index)
      : dipu::DIPUGeneratorImpl(device_index) {}

  void set_state(const c10::TensorImpl& state) override {
    at::detail::check_rng_state(state);
    auto state_size = state.numel();
    TORCH_CHECK(
        state_size == total_size || state_size == total_size - offset_size,
        "RNG state is wrong size");

    at::Tensor state_tmp(
        state.shallow_copy_and_detach(state.version_counter(), true));
    state_ = state_tmp;
    state_need_reset_ = false;
  }

  void update_state() const override {
    if (state_need_reset_) {
      state_ = at::detail::empty_cpu({(int64_t)total_size},
                                     c10::ScalarType::Byte, c10::nullopt,
                                     c10::nullopt, c10::nullopt, c10::nullopt);
      auto rng_state = state_.data_ptr<uint8_t>();
      uint64_t seed = this->current_seed();
      int64_t offset = 0;
      memcpy(rng_state, &seed, seed_size);
      memcpy(rng_state + seed_size, &offset, offset_size);
      state_need_reset_ = false;
    }
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<NPUGeneratorImpl>(device_index);
}

}  // namespace dipu
