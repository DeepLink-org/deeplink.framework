#include <cstring>

#include <ATen/Functions.h>
#include <ATen/Utils.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>
#include <csrc_dipu/runtime/core/DIPUGuard.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

namespace dipu {

class DROPLETGeneratorImpl : public dipu::DIPUGeneratorImpl {
 private:
  static constexpr std::size_t seed_size = sizeof(uint64_t);
  static constexpr std::size_t offset_size = sizeof(int64_t);
  static constexpr std::size_t total_size = seed_size + offset_size;

 public:
  explicit DROPLETGeneratorImpl(at::DeviceIndex device_index)
      : dipu::DIPUGeneratorImpl(device_index) {}

  void set_state(const c10::TensorImpl& state) override {
    at::detail::check_rng_state(state);
    auto state_size = state.numel();
    TORCH_CHECK(
        state_size == total_size || state_size == total_size - offset_size,
        "RNG state size is invalid");

    state_ = at::Tensor(
        state.shallow_copy_and_detach(state.version_counter(), true));
    state_need_reset_ = false;
  }

  void update_state() const override {
    if (state_need_reset_) {
      state_ = at::detail::empty_cpu({static_cast<int64_t>(total_size)},
                                     c10::ScalarType::Byte, c10::nullopt,
                                     c10::nullopt, c10::nullopt, c10::nullopt);
      auto rng_state = state_.data_ptr<uint8_t>();
      uint64_t seed = this->current_seed();

      std::memcpy(rng_state, &seed, seed_size);
      std::memset(rng_state + seed_size, 0, offset_size);
      state_need_reset_ = false;
    }
  }
};

// NOLINTNEXTLINE(readability-const-return-type)
const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<DROPLETGeneratorImpl>(device_index);
}

}  // namespace dipu
