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

inline std::string tensor_to_one_line_string(const at::Tensor& tensor) {
  /*
   * This function retrieves the built-in string representation of the input
   * tensor from PyTorch, and then simply flattens it into a single-line format.
   * For example: (0.01 * (-9.7712, -7.9712, -7.1297)) where 0.01 is a scale
   * generated from PyTorch.
   */
  std::ostringstream stream;
  stream << tensor;
  std::string raw_string = stream.str();
  std::string result_string = "(";
  result_string.reserve(2 * raw_string.size());
  bool is_scale_printed = false;
  bool is_first_line = true;
  for (char c : raw_string) {
    if ('\n' == c) {
      if (is_scale_printed && is_first_line) {
        is_first_line = false;
        continue;
      }
      is_first_line = false;
      result_string += ", ";
    } else if ('*' == c) {
      is_scale_printed = true;
      result_string += " * (";
    } else if (' ' == c) {
      continue;
    } else if ('[' == c) {
      break;
    } else {
      result_string += c;
    }
  }
  size_t result_size = result_string.size();
  if (result_size >= 2 && ',' == result_string[result_size - 2] &&
      ' ' == result_string[result_size - 1]) {
    result_string.erase(result_size - 2);
  }
  if (is_scale_printed) {
    result_string += ')';
  }
  result_string += ')';
  return result_string;
}

inline std::string allclose_autocompare(const at::Tensor& tensor_cpu,
                                        const at::Tensor& tensor_device,
                                        int indentation = 2) {
  std::ostringstream stream;
  stream << std::setfill(' ');
  if (tensor_cpu.defined() && tensor_device.defined()) {
    try {
      constexpr double tolerance_absolute = 1e-4;
      constexpr double tolerance_relative = 1e-5;
      const at::Tensor& tensor_cpu_from_device =
          to_cpu_without_diopi(tensor_device);
      bool passed = at::allclose(tensor_cpu, tensor_cpu_from_device,
                                 tolerance_absolute, tolerance_relative, true);
      if (passed) {
        stream << std::setw(indentation) << ""
               << "allclose"
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_cpu:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensor_cpu)
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_device:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensor_device);
      } else {
        auto diff = at::abs(tensor_cpu - tensor_cpu_from_device);
        auto mae = diff.mean().item<double>();
        auto max_diff = diff.max().item<double>();
        constexpr int printing_count = 10;
        stream << std::setw(indentation) << ""
               << "not_close, max diff: " << max_diff << ", MAE: " << mae
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_cpu:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensor_cpu)
               << "\n"
               << std::setw(indentation + 2) << ""
               << "First " << printing_count << " values or fewer:"
               << "\n"
               << std::setw(indentation + 2) << ""
               << tensor_to_one_line_string(
                      tensor_cpu.flatten().slice(0, 0, printing_count))
               << "\n"
               << std::setw(indentation) << ""
               << "tensor_device:"
               << "\n"
               << std::setw(indentation + 2) << "" << dumpArg(tensor_device)
               << "\n"
               << std::setw(indentation + 2) << ""
               << "First " << printing_count << " values or fewer:"
               << "\n"
               << std::setw(indentation + 2) << ""
               << tensor_to_one_line_string(
                      tensor_cpu_from_device.flatten().slice(0, 0,
                                                             printing_count));
      }
    } catch (...) {
      stream << std::setw(indentation) << ""
             << "compare_error: not_close";
    }
  } else {
    if (tensor_cpu.defined() != tensor_device.defined()) {
      stream << std::setw(indentation) << ""
             << "not_close: one of (tensor_cpu, tensor_device) is undefined, "
                "while the other is defined"
             << "\n"
             << std::setw(indentation) << ""
             << "tensor_cpu:"
             << "\n"
             << std::setw(indentation + 2) << "" << dumpArg(tensor_cpu) << "\n"
             << std::setw(indentation) << ""
             << "tensor_device:"
             << "\n"
             << std::setw(indentation + 2) << "" << dumpArg(tensor_device);
    } else {
      stream << std::setw(indentation) << ""
             << "allclose: both of (tensor_cpu, tensor_device) are undefined";
    }
  }
  return stream.str();
}

inline std::string allclose_autocompare(
    const c10::ArrayRef<at::Tensor>& tensor_list_cpu,
    const c10::ArrayRef<at::Tensor>& tensor_list_device, int indentation = 2) {
  std::ostringstream stream;
  stream << std::setfill(' ');
  if (tensor_list_cpu.size() != tensor_list_device.size()) {
    stream << std::setw(indentation) << ""
           << "not_allclose: "
           << "tensor_list_cpu has " << tensor_list_cpu.size()
           << "tensors, while tensor_list_device has "
           << tensor_list_device.size() << "tensors";
  } else {
    for (size_t i = 0; i < tensor_list_cpu.size(); ++i) {
      stream << std::setw(indentation) << "" << i << "-th:"
             << "\n"
             << allclose_autocompare(tensor_list_cpu[i], tensor_list_device[i],
                                     indentation + 2);
      if (i < tensor_list_cpu.size() - 1) {
        stream << "\n";
      }
    }
  }
  return stream.str();
}

}  // namespace native
}  // namespace dipu
