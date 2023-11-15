// Copyright (c) 2023, DeepLink.
#include <ATen/Utils.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

namespace dipu {

static const size_t seed_size = sizeof(uint64_t);
static const size_t offset_size = sizeof(int64_t);
static const size_t total_size = seed_size + offset_size;

class TopsGeneratorImpl : public dipu::DIPUGeneratorImpl {
 protected:
  mutable std::once_flag init_state_flag;

 public:
  TopsGeneratorImpl(at::DeviceIndex device_index)
      : dipu::DIPUGeneratorImpl(device_index) {}

  void set_state(const c10::TensorImpl &new_state) override {
    at::detail::check_rng_state(new_state);
    auto new_state_size = new_state.numel();
    TORCH_CHECK(new_state_size == total_size ||
                    new_state_size == total_size - offset_size,
                "RNG state is wrong size");

    at::Tensor state_tmp(
        new_state.shallow_copy_and_detach(new_state.version_counter(), true));
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
      std::memcpy(rng_state, &seed, seed_size);
      std::memcpy(rng_state + seed_size, &offset, offset_size);
      state_need_reset_ = false;
    }
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<TopsGeneratorImpl>(device_index);
}

}  // namespace dipu
