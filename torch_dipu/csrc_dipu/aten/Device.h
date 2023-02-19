# pragma once

#include <ATen/ATen.h>

namespace dipu {

static constexpr c10::DeviceType NativeDeviceType = c10::DeviceType::CUDA;
static constexpr c10::DispatchKey NativeDispatchKey = c10::DispatchKey::CUDA;
static constexpr c10::DispatchKey NativeAutogradDispatchKey = c10::DispatchKey::AutogradCUDA;
static constexpr c10::Backend NativeBackend = c10::Backend::CUDA;
static const std::string dipu_device_str = "dipu";
static const std::string default_device_str = "cuda";

}  // namespace dipu