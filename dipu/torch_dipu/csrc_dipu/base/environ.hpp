// Copyright (c) 2024, DeepLink.
//
// dipu::environ contains all DIPU configurations set via environment variables.
//
// TODO(lljbash): move all env vars to this file.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "csrc_dipu/runtime/device/basedef.h"

namespace dipu::environ {

namespace detail {

template <typename T, typename U>
T getEnvOrDefault(const char* env_var, U&& default_value,
                  const char* type_name = typeid(T).name()) {
  // PERF: not considering performance here as this is only used on startup
  //       and not using <charconv> for backward-compatibility
  static_assert(std::is_convertible_v<U, T>);
  static_assert(!std::is_pointer_v<T> && !std::is_reference_v<T>);
  const char* env_cstr = std::getenv(env_var);
  if (!env_cstr) {
    return std::forward<U>(default_value);
  }
  if constexpr (std::is_same_v<T, std::string>) {
    return env_cstr;
  } else if constexpr (std::is_same_v<T, bool>) {
    // CMake-like boolean values
    std::string env_str(env_cstr);
    std::transform(env_str.begin(), env_str.end(), env_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (env_str == "1" || env_str == "true" || env_str == "on" ||
        env_str == "yes" || env_str == "y") {
      return true;
    }
    if (env_str == "0" || env_str == "false" || env_str == "off" ||
        env_str == "no" || env_str == "n") {
      return false;
    }
    return getEnvOrDefault<std::int64_t>(env_var, default_value, type_name);
  } else {
    std::istringstream env_ss(env_cstr);
    T value{};
    env_ss >> value;
    if (env_ss.fail()) {
      DIPU_LOGE(
          "Failed to parse env var %s='%s' as type '%s', using default value "
          "'%s'.",
          env_var, env_cstr, type_name, std::to_string(default_value).c_str());
      return std::forward<U>(default_value);
    }
    if (!env_ss.eof()) {
      DIPU_LOGW(
          "Only parsed %s out of %zu characters from env var %s='%s' as type "
          "'%s', using value '%s'.",
          std::to_string(env_ss.tellg()).c_str(), env_ss.str().size(), env_var,
          env_cstr, type_name, std::to_string(value).c_str());
    }
    return value;
  }
}

}  // namespace detail

#ifdef DIPU_ENV_VAR
#error "DIPU_ENV_VAR already defined."
#endif
#define DIPU_ENV_VAR(ACCESSOR, NAME, TYPE, DEFAULT)                          \
  inline const auto& ACCESSOR() {                                            \
    static auto value = detail::getEnvOrDefault<TYPE>(NAME, DEFAULT, #TYPE); \
    return value;                                                            \
  }

// Determine whether DipuOpRegister should register ops immediately in
// registerOpMaybeDelayed(), or delay the registration of ops until
// applyDelayedRegister() is called.
DIPU_ENV_VAR(immediateRegisterOp, "DIPU_IMMEDIATE_REGISTER_OP", bool, false);
const std::string kTorchAllocatorName = "TORCH";
DIPU_ENV_VAR(dipuHostMemCachingAlgorithm, "DIPU_HOST_MEMCACHING_ALGORITHM",
             std::string, kTorchAllocatorName);
DIPU_ENV_VAR(dipuDeviceMemCachingAlgorithm, "DIPU_DEVICE_MEMCACHING_ALGORITHM",
             std::string, kTorchAllocatorName);
DIPU_ENV_VAR(torchAllocatorConf, "TORCH_ALLOCATOR_CONF", std::string, "");

#undef DIPU_ENV_VAR

}  // namespace dipu::environ
