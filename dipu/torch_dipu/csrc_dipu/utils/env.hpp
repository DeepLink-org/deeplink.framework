// Copyright (c) 2024, DeepLink.
#pragma once
#include <sstream>

namespace dipu {

template <typename T>
T get_env_or_default(const char* env_name, const T& default_value) {
  const char* env = std::getenv(env_name);
  if (env == nullptr) {
    return default_value;
  }
  T value = default_value;
  std::istringstream(env) >> value;
  return value;
}

}  // namespace dipu
