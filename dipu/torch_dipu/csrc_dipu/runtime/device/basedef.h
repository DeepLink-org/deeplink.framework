// Copyright (c) 2023, DeepLink.
#pragma once

// todo:: dev api will remove pytorch dependency
#include <c10/core/Device.h>

#include <csrc_dipu/base/basedef.h>

// todo: move out deice dir to diopi
namespace dipu {

#define DIPU_API __attribute__((visibility("default")))

#define DIPU_WEAK __attribute__((weak))

// "default", "hidden", "protected" or "internal
#define DIPU_HIDDEN __attribute__((visibility("hidden")))

using enum_t = int32_t;

#define DIPU_STRING(x) #x
#define DIPU_CODELOC __FILE__ " (" DIPU_STRING(__LINE__) ")"

#define DIPU_LOGE(fmt, ...)                                              \
  printf("[ERROR]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)

#define DIPU_LOGW(fmt, ...)                                             \
  printf("[WARN]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)

namespace devapis {

enum class VendorDeviceType : enum_t {
  MLU,      // camb
  NPU,      // ascend
  CUDA,     // cuda
  GCU,      // gcu
  SUPA,     // Biren
  DROPLET,  // droplet
};

enum class EventStatus : enum_t { PENDING, RUNNING, DEFERRED, READY };

enum class OpStatus : enum_t {
  SUCCESS,
  ERR_UNKNOWN,
  ERR_NOMEM,
};

enum class MemCPKind : enum_t {
  D2H,
  H2D,
  D2D,
};

enum diclResult_t {
  /*! The operation was successful. */
  DICL_SUCCESS = 0x0,

  /*! undefined error */
  DICL_ERR_UNDEF = 0x01000,

};

struct DIPUDeviceStatus {
  size_t freeGlobalMem = 0;
};

struct DIPUDeviceProperties {
  std::string name;
  size_t totalGlobalMem = 0;
  int32_t major = 0;
  int32_t minor = 0;
  int32_t multiProcessorCount = 0;
};

using deviceId_t = c10::DeviceIndex;

}  // end namespace devapis
}  // end namespace dipu
