#include <ATen/Functions.h>
#include <ATen/Utils.h>

#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>
#include <csrc_dipu/runtime/core/DIPUGuard.h>
#include <csrc_dipu/runtime/device/deviceapis.h>

namespace dipu {

// Discriminate floating device type.
// static bool is_floating_device = true;

// just an example
// not implemented now
class KLXGeneratorImpl : public dipu::DIPUGeneratorImpl {
 public:
  KLXGeneratorImpl(at::DeviceIndex device_index)
      : dipu::DIPUGeneratorImpl(device_index) {}

  void set_state(const c10::TensorImpl& state) override {}

  void update_state() const override {}
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<KLXGeneratorImpl>(device_index);
}

}  // namespace dipu
