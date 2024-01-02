#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Generator.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/allclose.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/string_view.h>

#include "csrc_dipu/runtime/core/DIPUStream.h"
#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/Log.h>

namespace dipu {
namespace native {

inline bool checkTensorDevice() {
  bool enable = []() {
    const char* env_ptr = std::getenv("DIPU_CHECK_TENSOR_DEVICE");
    if (env_ptr == nullptr) {
      return false;
    }
    return std::atoi(env_ptr) > 0;
  }();
  return enable;
}

inline void synchronizeIfEnable() {
  const char* mode = std::getenv("DIPU_SYNC_EXEC_MODE");
  if (mode != nullptr) {
    DIPU_LOG_ONCE << "The synchronous operation is performed after "
                  << "the diopi function call because the DIPU_SYNC_EXEC_MODE "
                     "environment variable is set"
                  << '\n';
    dipu::getCurrentDIPUStream().synchronize();
  }
}

inline int dumpOpArgLevel() {
  const char* env_ptr = std::getenv("DIPU_DUMP_OP_ARGS");
  int level = env_ptr ? std::atoi(env_ptr) : 0;
  return level;
}

template <typename T>
std::string dumpArg(const T& t) {
  std::stringstream stream;
  stream << t;
  return stream.str();
}

template <typename T1>
std::string dumpArg(const c10::optional<T1>& opt_t) {
  std::stringstream stream;
  if (opt_t.has_value()) {
    stream << dumpArg(opt_t.value());
  }
  return stream.str();
}

template <typename T>
std::string dumpArg(const c10::OptionalArrayRef<T>& opt_t) {
  std::stringstream stream;
  if (opt_t.has_value()) {
    stream << dumpArg(opt_t.value());
  }
  return stream.str();
}

template <typename T1, template <typename elem> class container>
std::string dumpArg(const container<T1>& t) {
  std::stringstream stream;
  for (auto iter = t.begin(); iter != t.end(); ++iter) {
    stream << dumpArg(*iter) << ", ";
  }
  return stream.str();
}

template <>
inline std::string dumpArg(const at::Tensor& tensor) {
  std::stringstream stream;
  if (tensor.defined()) {
    stream << "numel: " << tensor.numel() << ", sizes: " << tensor.sizes()
           << ", stride: " << tensor.strides()
           << ", is_view: " << tensor.is_view() << ", dtype: " << tensor.dtype()
           << ", device:" << tensor.device() << ", layout:" << tensor.layout()
           << ", requires_grad: " << (tensor.requires_grad() ? "true" : "false")
           << ", pinned_memory: " << (tensor.is_pinned() ? "true" : "false")
           << ", memory_format: " << tensor.suggest_memory_format()
           << ",  data_ptr: " << tensor.data_ptr();
    if (dumpOpArgLevel() > 2) {
      stream << '\n' << tensor;
    }
  } else {
    stream << "undefined";
  }
  return stream.str();
}

template <>
inline std::string dumpArg(const at::Scalar& t) {
  std::stringstream stream;
  stream << t;
  return stream.str();
}

template <>
inline std::string dumpArg(const c10::string_view& t) {
  return dumpArg(std::string(t.data()));
}

template <>
inline std::string dumpArg(const at::Generator& t) {
  return "";
}

template <typename T, size_t N>
std::string dumpArg(const std::array<T, N>& t) {
  std::stringstream stream;
  for (auto iter = t.begin(); iter != t.end(); ++iter) {
    stream << dumpArg(*iter) << " ";
  }
  return stream.str();
}

template <>
inline std::string dumpArg(const c10::List<c10::optional<at::Tensor>>& t) {
  std::stringstream stream;
  stream << "size:" << t.size() << '\n';
  for (int i = 0; i < t.size(); ++i) {
    bool has_value = t[i].has_value();
    stream << "\t" << i << "th: has_value:" << has_value << " ";
    if (has_value) {
      stream << dumpArg(t[i].value());
    }
    stream << '\n';
  }
  return stream.str();
}

template <typename T1, typename T2, template <typename elem1> class container1,
          template <typename elem2> class container2>
std::vector<int64_t> infer_reduce_op_shape(const container1<T1>& input_shape,
                                           const container2<T2>& dims,
                                           bool keepdim) {
  if (dims.size() <= 0) {
    return {};
  }
  if (keepdim) {
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    for (auto iter = dims.begin(); iter != dims.end(); ++iter) {
      auto dim = *iter;
      dim += dim < 0 ? input_shape.size() : 0;
      output_shape[dim] = 1;
    }
    return output_shape;
  }
  std::vector<int64_t> output_shape;
  output_shape.reserve(input_shape.size() - dims.size());
  for (int i = 0; i < input_shape.size(); ++i) {
    bool reduce_dim = false;
    for (auto iter = dims.begin(); iter != dims.end(); ++iter) {
      auto dim = *iter;
      dim += dim < 0 ? input_shape.size() : 0;
      if (dim == i) {
        reduce_dim = true;
        break;
      }
    }
    if (!reduce_dim) {
      output_shape.push_back(input_shape.at(i));
    }
  }
  return output_shape;
}

inline std::string _allclose(const at::Tensor& a, const at::Tensor& b) {
  if (a.defined() && b.defined()) {
    try {
      constexpr double tolerance_absolute = 1e-4;
      constexpr double tolerance_relative = 1e-5;
      if (at::allclose(a.cpu(), b.cpu(), tolerance_absolute, tolerance_relative,
                       true)) {
        return "allclose";
      }
      auto diff = at::abs(a.cpu() - b.cpu());
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

inline std::string _allclose(const c10::ArrayRef<at::Tensor>& a,
                             const c10::ArrayRef<at::Tensor>& b) {
  if (a.size() != b.size()) {
    return "not_allclose:";
  }
  std::string result;
  for (size_t i = 0; i < a.size(); ++i) {
    result += std::to_string(i) + "th " + _allclose(a[i], b[i]) + "; ";
  }
  return result;
}

}  // namespace native
}  // namespace dipu
