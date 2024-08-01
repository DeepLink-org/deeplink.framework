// Copyright (c) 2024, DeepLink.
#pragma once

#include <variant>

#if DIPU_TORCH_VERSION >= 20200
#include <c10/util/ApproximateClock.h>
#else
#include <c10/util/variant.h>
#include <torch/csrc/profiler/util.h>
#endif
#include <c10/util/C++17.h>

namespace dipu {
namespace profile {

#if DIPU_TORCH_VERSION >= 20200
using c10::approx_time_t;
using c10::ApproximateClockToUnixTimeConverter;
using c10::getApproximateTime;
using std::get;
using std::holds_alternative;
using std::visit;
#else
using c10::get;
using c10::holds_alternative;
using c10::visit;
using torch::profiler::impl::approx_time_t;
using torch::profiler::impl::ApproximateClockToUnixTimeConverter;
using torch::profiler::impl::getApproximateTime;
#endif

inline time_t torchGetTime() {
#if DIPU_TORCH_VERSION >= 20200
  return c10::getTime();
#else
  return torch::profiler::impl::getTime();
#endif
}

inline int64_t getTimeUs() {
  auto constexpr scale = int64_t{1000};
  return torchGetTime() / scale;
}

}  // namespace profile
}  // namespace dipu
