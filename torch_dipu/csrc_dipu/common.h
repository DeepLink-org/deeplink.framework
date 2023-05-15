#pragma once

#include <ATen/Utils.h>
#include <c10/core/DispatchKey.h>

#define DIPU_API __attribute__ ((visibility ("default")))

#define DIPU_WEAK  __attribute__((weak))

// "default", "hidden", "protected" or "internal
#define DIPU_HIDDEN __attribute__ ((visibility ("hidden")))


typedef int32_t enum_t;

#define C10_COMPILE_TIME_MAX_DIPUS 16

#define DIPU_STRING(x) #x
#define DIPU_CODELOC __FILE__ " (" DIPU_STRING(__LINE__) ")"


#define DIPU_LOGE(fmt, ...)          \
  printf(                           \
      "[ERROR]%s,%s:%u:" #fmt "\n", \
      __FUNCTION__,                 \
      __FILE__,                     \
      __LINE__,                     \
      ##__VA_ARGS__)

#define DIPU_LOGW(fmt, ...)         \
  printf(                          \
      "[WARN]%s,%s:%u:" #fmt "\n", \
      __FUNCTION__,                \
      __FILE__,                    \
      __LINE__,                    \
      ##__VA_ARGS__)

namespace dipu {
namespace devapis {

enum class VendorDeviceType : enum_t {
  MLU,  //camb
  NPU,  //ascend
  CUDA, //cuda
  GCU,  //gcu
};

constexpr const char* VendorTypeToStr(VendorDeviceType t) noexcept {
  switch (t) {
    case VendorDeviceType::MLU: return "MLU";
    case VendorDeviceType::CUDA: return "CUDA";
    case VendorDeviceType::NPU: return "NPU";
    case VendorDeviceType::GCU: return "GCU";
  }
}

enum class EventStatus: enum_t {
  PENDING,
  RUNNING,
  DEFERRED,
  READY
};

enum class OpStatus: enum_t {
  SUCCESS,
  ERR_UNKNOWN,
  ERR_NOMEM,
};

enum class MemCPKind: enum_t {
  D2H, 
  H2D,
  D2D,
};

} // end ns devapis

const auto DIPU_DEVICE_TYPE = at::DeviceType::PrivateUse1;

const auto DIPU_DISPATCH_KEY = c10::DispatchKey::PrivateUse1;
const auto DIPU_DISPATCH_AUTOGRAD_KEY = c10::DispatchKey::AutogradPrivateUse1;


const auto DIPU_Backend_TYPE = c10::Backend::PrivateUse1;

extern devapis::VendorDeviceType VENDOR_TYPE;

DIPU_API bool isDeviceTensor(const at::Tensor &tensor);

} // end ns dipu

#define DIPU_DEVICE_TYPE_MACRO PrivateUse1
#define DIPU_AUTOGRAD_DEVICE_TYPE_MACRO AutogradPrivateUse1

