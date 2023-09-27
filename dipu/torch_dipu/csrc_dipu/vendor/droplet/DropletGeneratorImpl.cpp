#include <ATen/Utils.h>
#include <ATen/Functions.h>

#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>
#include <csrc_dipu/runtime/core/DIPUGuard.h>
#include <csrc_dipu/runtime/core/DIPUGeneratorImpl.h>

namespace dipu {
  static deviceHandle_t getDeviceHandler(c10::DeviceIndex device_index) {
  if (device_index == -1) {
    device_index = devapis::current_device();
  }
  deviceHandle_t handle;
  // cnnlCreate(&handle);
  auto stream = getCurrentDIPUStream(device_index);
  // cnnlSetQueue(handle, stream.rawstream());
  return handle;
}

// Discriminate floating device type.
// static bool is_floating_device = true;

// just an example
// not implemented now
class DROPLETGeneratorImpl : public dipu::DIPUGeneratorImpl {
public:
  DROPLETGeneratorImpl(at::DeviceIndex device_index): dipu::DIPUGeneratorImpl(device_index) {
  }

  void init_state() const override {
  }

  void set_state(const c10::TensorImpl& state) override {
  }

  void update_state() const override {
  }
};

const at::Generator vendorMakeGenerator(at::DeviceIndex device_index) {
  return at::make_generator<DROPLETGeneratorImpl>(device_index);
}

}