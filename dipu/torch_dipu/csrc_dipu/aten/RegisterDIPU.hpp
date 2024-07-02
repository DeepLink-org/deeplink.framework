// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/core/boxing/BoxedKernel.h>
#include <ATen/core/stack.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "OpRegister.hpp"  // IWYU pragma: export
#include "ops/OpRegexMatch.hpp"

namespace at {

void dipu_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys,
                   torch::jit::Stack* stack);

}  // namespace at

// Print the warning message only once for one process.
// NOLINTBEGIN(bugprone-macro-parentheses): x cannot be in parentheses
#define DIPU_LOG_WARNING_ONCE(x)     \
  do {                               \
    static bool should_print = true; \
    if (should_print) {              \
      std::cout << x;                \
      should_print = false;          \
    }                                \
  } while (0)
// NOLINTEND(bugprone-macro-parentheses)

// Check the environment variable and call the DIPU_LOG_WARNING_ONCE
#define DIPU_OP_LOG_WARNING_ONCE(...)                       \
  do {                                                      \
    const char* env = std::getenv("DIPU_DUMP_OP_ARGS");     \
    int env_value = (env != nullptr) ? std::atoi(env) : -1; \
    if (env_value >= 0) {                                   \
      DIPU_LOG_WARNING_ONCE(__VA_ARGS__);                   \
    }                                                       \
  } while (0)

// Temporarily not implement 'sub-dispatch from box' (from torch box func ->
// ourself unbox func) which described in design doc. because: 1. it need many
// add type trait code. 2. pytorch seems are sorting out infer and other
// pre/post code. so we shouldn't created a new preprocess logic?
// so just do a simple runtime cpu fallback to support diopi func loss

// It mat be necessary to determine whether to keep torchop default impl
// for non-custom ops through function dipuKeepTorchopDefaultImpl firstly in the
// future, and we use force fallback to keep torchop default impl now.
#define NO_CUSTOMFALLBACK_WITH_AUTOCOMPARE_REGISTER(opname, diopiFunc,         \
                                                    wrapperFunc)               \
  do {                                                                         \
    if ((reinterpret_cast<void*>(diopiFunc) != nullptr) &&                     \
        (!dipu::op_regex_match::isOpMatch(                                     \
            opname, dipu::op_regex_match::kFallbackMatchers))) {               \
      if (dipu::op_regex_match::isOpMatch(                                     \
              opname, dipu::op_regex_match::kAutocompareMatchers)) {           \
        m.impl(opname, TORCH_FN(wrapperFunc##_autocompare));                   \
      } else {                                                                 \
        m.impl(opname, TORCH_FN(wrapperFunc));                                 \
      }                                                                        \
    } else {                                                                   \
      if ((reinterpret_cast<void*>(diopiFunc) == nullptr)) {                   \
        DIPU_OP_LOG_WARNING_ONCE(#diopiFunc << " is not yet implemented, "     \
                                            << (opname)                        \
                                            << " will be fallback to cpu\n");  \
      } else {                                                                 \
        DIPU_OP_LOG_WARNING_ONCE("force fallback has been set, "               \
                                 << (opname) << " will be fallback to cpu\n"); \
      }                                                                        \
    }                                                                          \
  } while (false);

#define NO_CUSTOMFALLBACK_NO_AUTOCOMPARE_REGISTER(opname, diopiFunc,           \
                                                  wrapperFunc)                 \
  do {                                                                         \
    if ((reinterpret_cast<void*>(diopiFunc) != nullptr) &&                     \
        (!dipu::op_regex_match::isOpMatch(                                     \
            opname, dipu::op_regex_match::kFallbackMatchers))) {               \
      m.impl(opname, TORCH_FN(wrapperFunc));                                   \
    } else {                                                                   \
      if ((reinterpret_cast<void*>(diopiFunc) == nullptr)) {                   \
        DIPU_OP_LOG_WARNING_ONCE(#diopiFunc << " is not yet implemented, "     \
                                            << (opname)                        \
                                            << " will be fallback to cpu\n");  \
      } else {                                                                 \
        DIPU_OP_LOG_WARNING_ONCE("force fallback has been set, "               \
                                 << (opname) << " will be fallback to cpu\n"); \
      }                                                                        \
    }                                                                          \
  } while (false);

// Determine whether to keep torchop default impl for custom ops through
// function dipuKeepTorchopDefaultImpl firstly.
#define WITH_CUSTOMFALLBACK_WITH_AUTOCOMPARE_REGISTER(                         \
    opname, diopi_func, force_fallback, wrapper_func, custom_fallback_func)    \
  do {                                                                         \
    if (dipu::native::dipuKeepTorchopDefaultImpl(opname)) {                    \
      break;                                                                   \
    }                                                                          \
    if ((reinterpret_cast<void*>(diopi_func) != nullptr) &&                    \
        !((force_fallback) ||                                                  \
          dipu::op_regex_match::isOpMatch(                                     \
              opname, dipu::op_regex_match::kFallbackMatchers))) {             \
      if (dipu::op_regex_match::isOpMatch(                                     \
              opname, dipu::op_regex_match::kAutocompareMatchers)) {           \
        m.impl(opname, TORCH_FN(wrapper_func##_autocompare));                  \
      } else {                                                                 \
        m.impl(opname, TORCH_FN(wrapper_func));                                \
      }                                                                        \
    } else {                                                                   \
      if ((reinterpret_cast<void*>(diopi_func) == nullptr)) {                  \
        DIPU_OP_LOG_WARNING_ONCE(#diopi_func << " is not yet implemented, "    \
                                             << (opname)                       \
                                             << " will be fallback to cpu\n"); \
      } else {                                                                 \
        DIPU_OP_LOG_WARNING_ONCE("force fallback has been set, "               \
                                 << (opname) << " will be fallback to cpu\n"); \
      }                                                                        \
      m.impl(opname, TORCH_FN(custom_fallback_func));                          \
    }                                                                          \
  } while (false);

#define WITH_CUSTOMFALLBACK_NO_AUTOCOMPARE_REGISTER(                           \
    opname, diopi_func, force_fallback, wrapper_func, custom_fallback_func)    \
  do {                                                                         \
    if (dipu::native::dipuKeepTorchopDefaultImpl(opname)) {                    \
      break;                                                                   \
    }                                                                          \
    if ((reinterpret_cast<void*>(diopi_func) != nullptr) &&                    \
        !((force_fallback) ||                                                  \
          dipu::op_regex_match::isOpMatch(                                     \
              opname, dipu::op_regex_match::kFallbackMatchers))) {             \
      m.impl(opname, TORCH_FN(wrapper_func));                                  \
    } else {                                                                   \
      if ((reinterpret_cast<void*>(diopi_func) == nullptr)) {                  \
        DIPU_OP_LOG_WARNING_ONCE(#diopi_func << " is not yet implemented, "    \
                                             << (opname)                       \
                                             << " will be fallback to cpu\n"); \
      } else {                                                                 \
        DIPU_OP_LOG_WARNING_ONCE("force fallback has been set, "               \
                                 << (opname) << " will be fallback to cpu\n"); \
      }                                                                        \
      m.impl(opname, TORCH_FN(custom_fallback_func));                          \
    }                                                                          \
  } while (false);

#define DIPU_LIBRARY_IMPL(ns, k, m) _DIPU_LIBRARY_IMPL(ns, k, m, C10_UID)

#define _DIPU_LIBRARY_IMPL(ns, k, m, uid)                                \
  static void C10_CONCATENATE(DIPU_LIBRARY_IMPL_init_##ns##_##k##_,      \
                              uid)(torch::Library&);                     \
  ::at::DipuOpRegisterHelper C10_CONCATENATE(                            \
      DIPU_LIBRARY_IMPL_static_init_##ns##_##k##_,                       \
      C10_CONCATENATE(__LINE__, uid))(                                   \
      (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::k)      \
           ? &C10_CONCATENATE(DIPU_LIBRARY_IMPL_init_##ns##_##k##_, uid) \
           : [](torch::Library&) -> void {}),                            \
      #ns, c10::make_optional(c10::DispatchKey::k), __FILE__, __LINE__); \
  void C10_CONCATENATE(DIPU_LIBRARY_IMPL_init_##ns##_##k##_,             \
                       uid)(torch::Library & (m))
