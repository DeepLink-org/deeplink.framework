#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include <ATen/core/TensorBody.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/allclose.h>
#include <c10/util/ArrayRef.h>

#include "csrc_dipu/aten/ops/DIPUCopy.hpp"
#include "csrc_dipu/runtime/device/deviceapis.h"

namespace dipu {
namespace native {

inline at::Tensor to_cpu_without_diopi(const at::Tensor& in) {
  if (in.is_cpu()) {
    return in;
  }

  at::Tensor out = at::empty_strided(in.sizes(), in.strides(),
                                     in.options().device(c10::Device("cpu")));
  if (in.nbytes() > 0) {
    dipu::devapis::memCopyD2H(in.nbytes(), out.data_ptr(), in.data_ptr());
  }
  return out;
}

inline std::string allclose_autocompare(const at::Tensor& tensorCpu,
                                        const at::Tensor& tensorDevice,
                                        int indentation = 2) {
  std::ostringstream stream;
  stream << std::setfill(' ');
  if (tensorCpu.defined() && tensorDevice.defined()) {
    try {
      constexpr double tolerance_absolute = 1e-4;
      constexpr double tolerance_relative = 1e-5;
      const at::Tensor& tensorCpuFromDevice =
          to_cpu_without_diopi(tensorDevice);
      bool passed = at::allclose(tensorCpu, tensorCpuFromDevice,
                                 tolerance_absolute, tolerance_relative, true);
      if (passed) {
        stream << std::setw(indentation) << ""
               << "allclose"
               << "\n"
               << std::setw(indentation) << ""
               << "--------------------"
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_cpu:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensorCpu) << "\n"
               << std::setw(indentation) << ""
               << "--------------------"
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_device:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensorDevice);
      } else {
        auto diff = at::abs(tensorCpu - tensorCpuFromDevice);
        auto mae = diff.mean().item<double>();
        auto max_diff = diff.max().item<double>();
        constexpr int printing_count = 10;
        stream << std::setw(indentation) << ""
               << "not_close, max diff: " << max_diff << ", MAE: " << mae
               << "\n"
               << std::setw(indentation) << ""
               << "--------------------"
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_cpu:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensorCpu) << "\n"
               << std::setw(indentation + 2) << ""
               << "First 10 values or fewer:"
               << "\n"
               << tensorCpu.flatten().slice(0, 0, printing_count) << "\n"
               << std::setw(indentation) << ""
               << "--------------------"
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_device:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensorDevice)
               << "\n"
               << std::setw(indentation + 2) << ""
               << "First 10 values or fewer:"
               << "\n"
               << tensorCpuFromDevice.flatten().slice(0, 0, printing_count)
               << "\n"
               << std::setw(indentation) << ""
               << "--------------------"
               << "\n"
               << std::setw(indentation) << ""
               << "diff(= tensor_cpu - tensor_device):"
               << "\n"
               << std::setw(indentation + 2) << ""
               << "First 10 values or fewer:"
               << "\n"
               << tensorCpu.flatten().slice(0, 0, printing_count) -
                      tensorCpuFromDevice.flatten().slice(0, 0, printing_count);
      }
    } catch (...) {
      stream << std::setw(indentation) << ""
             << "compare_error: not_close";
    }
  } else {
    if (tensorCpu.defined() != tensorDevice.defined()) {
      stream << std::setw(indentation) << ""
             << "not_close: one of (tensor_cpu, tensor_device) is undefined, "
                "while the other is defined"
             << "\n"
             << std::setw(indentation) << ""
             << "--------------------"
             << "\n"
             << std::setw(indentation) << ""
             << "tensor_cpu:"
             << "\n"
             << std::setw(indentation + 2) << "" << dumpArg(tensorCpu) << "\n"
             << std::setw(indentation) << ""
             << "--------------------"
             << "\n"
             << std::setw(indentation) << ""
             << "tensor_device:"
             << "\n"
             << std::setw(indentation + 2) << "" << dumpArg(tensorDevice);
    } else {
      stream << std::setw(indentation) << ""
             << "allclose: both of (tensor_cpu, tensor_device) are undefined";
    }
  }
  return stream.str();
}

inline std::string allclose_autocompare(
    const c10::ArrayRef<at::Tensor>& tensorListCpu,
    const c10::ArrayRef<at::Tensor>& tensorListDevice, int indentation = 2) {
  std::ostringstream stream;
  stream << std::setfill(' ');
  if (tensorListCpu.size() != tensorListDevice.size()) {
    stream << std::setw(indentation) << ""
           << "not_allclose: "
           << "tensor_list_cpu has " << tensorListCpu.size()
           << "tensors, while tensor_list_device has "
           << tensorListDevice.size() << "tensors";
  } else {
    for (size_t i = 0; i < tensorListCpu.size(); ++i) {
      stream << std::setw(indentation) << "" << i << "-th:"
             << "\n"
             << allclose_autocompare(tensorListCpu[i], tensorListDevice[i],
                                     indentation + 2);
      if (i < tensorListCpu.size() - 1) {
        stream << "\n";
      }
    }
  }
  return stream.str();
}

}  // namespace native
}  // namespace dipu
