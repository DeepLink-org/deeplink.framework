// Copyright (c) 2024, DeepLink.
//
// dipu::environ contains all DIPU configurations set via environment variables.
//
// TODO(lljbash): move all env vars to this file.

#pragma once

#include <charconv>
#include <cstdlib>
#include <string_view>

namespace dipu::environ {

namespace detail {

inline std::string_view getEnvOrEmpty(const char* env_var) {
  const char* env = std::getenv(env_var);
  return env ? env : "";
}

inline int getEnvIntOrDefault(const char* env_var, int default_value = 0) {
  int value = default_value;
  auto env = getEnvOrEmpty(env_var);
  std::from_chars(env.data(), env.data() + env.size(), value);
  return value;
}

inline bool getEnvFlag(const char* env_var) {
  return getEnvIntOrDefault(env_var, 0) > 0;
}

}  // namespace detail

// Determine whether DipuOpRegister should register ops immediately in
// registerOpMaybeDelayed(), or delay the registration of ops until
// applyDelayedRegister() is called.
inline bool immediateRegisterOp() {
  static bool flag = detail::getEnvFlag("DIPU_IMMEDIATE_REGISTER_OP");
  return flag;
}

}  // namespace dipu::environ
