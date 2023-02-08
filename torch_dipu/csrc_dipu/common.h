#pragma once

#include <ATen/Utils.h>
#include <c10/core/DispatchKey.h>

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

namespace torch_dipu {
namespace devapis {

enum class VendorDeviceType : enum_t {
  MLU,  //camb
  NPU,  //ascend
};

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

} // end ns devapis

namespace
{
const auto DIPU_DEVICE_TYPE = at::DeviceType::PrivateUse1;
#define DIPU_DEVICE_TYPE_MACRO PrivateUse1

const auto DIPU_DISPATCH_KEY = c10::DispatchKey::PrivateUse1;


const auto DIPU_Backend_TYPE = c10::Backend::PrivateUse1;


const extern devapis::VendorDeviceType VENDOR_TYPE;
} // end anonymous ns

} // end ns torch_dipu
