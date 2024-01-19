#pragma once

#include <string>

#include <ATen/core/TensorBody.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/allclose.h>
#include <c10/util/ArrayRef.h>

#include "csrc_dipu/aten/ops/DIPUCopy.hpp"

namespace dipu {
namespace native {

inline at::Tensor to_cpu_without_diopi(const at::Tensor& in) {
  if (in.is_cpu()) {
    return in;
  }

  at::Tensor out =
      at::empty(in.sizes(), in.scalar_type(), in.layout(), c10::Device("cpu"),
                false, in.suggest_memory_format());
  static dipu::DIPUCopyInpOnCPU _copy_op_without_diopi;
  _copy_op_without_diopi.run(out, in, false);
  return out;
}

inline std::string allclose_autocompare(const at::Tensor& a,
                                        const at::Tensor& b) {
  if (a.defined() && b.defined()) {
    try {
      constexpr double tolerance_absolute = 1e-4;
      constexpr double tolerance_relative = 1e-5;
      at::Tensor a_cpu = to_cpu_without_diopi(a);
      at::Tensor b_cpu = to_cpu_without_diopi(b);
      if (at::allclose(a_cpu, b_cpu, tolerance_absolute, tolerance_relative,
                       true)) {
        return "allclose";
      }
      auto diff = at::abs(a_cpu - b_cpu);
      auto mae = diff.mean().item<double>();
      auto max_diff = diff.max().item<double>();
      return "not_close, max diff: " + std::to_string(max_diff) +
             ", MAE: " + std::to_string(mae);
    } catch (...) {
      return "compare_error: not_close";
    }
  } else {
    if (a.defined() != b.defined()) {
      return "not_close, one of tensor inputs is empty";
    }
    return "allclose";
  }
}

inline std::string allclose_autocompare(const c10::ArrayRef<at::Tensor>& a,
                                        const c10::ArrayRef<at::Tensor>& b) {
  if (a.size() != b.size()) {
    return "not_allclose:";
  }
  std::string result;
  for (size_t i = 0; i < a.size(); ++i) {
    result +=
        std::to_string(i) + "th " + allclose_autocompare(a[i], b[i]) + "; ";
  }
  return result;
}

}  // namespace native
}  // namespace dipu
