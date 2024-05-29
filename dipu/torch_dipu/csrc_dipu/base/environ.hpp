// Copyright (c) 2024, DeepLink.
//
// dipu::environ contains all DIPU configurations set via environment variables.
//
// TODO(lljbash): move all env vars to this file.

#pragma once

#include <cstdlib>
#include <string_view>

// FIXME: DIPU is expected to be built with a compiler that supports C++17.
//        This is a temporary fix for older compilers that don't have
//        <charconv>. (e.g., GCC 7.5 in Acsend toolchain)
#if __has_include(<charconv>)
#include <charconv>
#else
#include <stdexcept>
#include <string>
#include <system_error>
namespace std {
// HACK: Define a subset of std::from_chars for compilers that don't support.
namespace my_from_chars {
struct from_chars_result {
  const char* ptr;
  std::errc ec;
};
inline from_chars_result from_chars(const char* first, const char* last,
                                    int& value, int base = 10) {  // NOLINT
  std::size_t pos{};
  std::errc ec{};
  try {
    value = std::stoi(std::string(first, last), &pos, base);
  } catch (const std::invalid_argument&) {
    ec = std::errc::invalid_argument;
  } catch (const std::out_of_range&) {
    // WARNING: pos is 0 in this case, which is not aligned with the standard.
    ec = std::errc::result_out_of_range;
  }
  return {first + pos, ec};
}
}  // namespace my_from_chars
using my_from_chars::from_chars;
using my_from_chars::from_chars_result;
}  // namespace std
#endif

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
  // TODO(lilingjie): maybe add warning here if conversion fails.
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
