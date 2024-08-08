// Copyright (c) 2023, DeepLink.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Generator.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/empty_strided.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/string_view.h>

#include "csrc_dipu/runtime/core/DIPUStream.h"
#include "csrc_dipu/runtime/device/deviceapis.h"
#include "csrc_dipu/runtime/rthelper.h"
#include "csrc_dipu/utils/Log.h"

namespace dipu {
namespace native {

// avoid infinite recursion when dumpArg() before calling diopiCopy()
inline at::Tensor tensor_clone_to_host(const at::Tensor& in) {
  if (in.is_cpu()) {
    return in;
  }

  auto opt = in.options().device(c10::Device("cpu"));
  auto out = at::empty_strided(in.sizes(), in.strides(), opt);
  if (in.nbytes() > 0) {
    dipu::getCurrentDIPUStream().synchronize();
    dipu::devapis::memCopyD2H(out.storage().nbytes(), out.data_ptr(),
                              in.data_ptr());
  }
  return out;
}

inline c10::optional<at::Tensor> tensor_clone_to_host(
    const c10::optional<at::Tensor>& in) {
  if (in) {
    if (auto& tensor = in.value(); tensor.defined()) {
      return c10::make_optional<at::Tensor>(tensor_clone_to_host(tensor));
    }
  }
  return c10::nullopt;
}

inline c10::List<c10::optional<at::Tensor>> tensor_clone_to_host(
    const c10::List<c10::optional<at::Tensor>>& in /* Tensor?[] */) {
  auto out = c10::List<c10::optional<at::Tensor>>();
  out.reserve(in.size());
  for (auto const& tensor : in) {
    out.push_back(tensor_clone_to_host(tensor));
  }
  return out;
}

template <typename R>
inline auto tensor_clone_to_host(const R& in)
    -> decltype(in.begin(), in.end(), std::vector<at::Tensor>()) {
  auto out = std::vector<at::Tensor>();
  out.reserve(in.size());
  for (auto const& tensor : in) {
    out.push_back(tensor_clone_to_host(tensor));
  }
  return out;
}

inline at::Tensor tensor_reference_or_clone_to_host(
    at::Tensor const& in,
    std::initializer_list<std::pair<at::Tensor const&, at::Tensor const&>>
        device_host_tensor_pairs) {
  for (auto const& [device, host] : device_host_tensor_pairs) {
    if (in.is_same(device)) {
      return host;
    }
  }
  return tensor_clone_to_host(in);
}

inline void tensor_copy_host_to_device(at::Tensor& out, const at::Tensor& in,
                                       DIPUStream stream) {
  if (out.is_cuda() && in.is_cpu()) {
    stream.synchronize();

    if (out.sizes() != in.sizes()) {
      auto device = out.options().device();
      auto option = in.options().device(device);
      out = at::empty_strided(in.sizes(), in.strides(), option);
    }

    auto size = out.storage().nbytes();
    dipu::devapis::memCopyH2D(size, out.data_ptr(), in.data_ptr());
  }
  // else may generate a warning or throw exception?
}

inline std::vector<at::Tensor> tensor_array_to_vector(
    at::ArrayRef<at::Tensor> in) {
  return in.vec();
}

// Warning: it returns reference, thus decltype(auto) is required to avoid copy.
inline std::vector<at::Tensor>& tensor_array_to_vector(
    std::vector<at::Tensor>& in) {
  return in;
}

inline bool checkTensorDevice() {
  static bool enable = []() {
    const char* env_ptr = std::getenv("DIPU_CHECK_TENSOR_DEVICE");
    if (env_ptr == nullptr) {
      return false;
    }
    return std::atoi(env_ptr) > 0;
  }();
  return enable;
}

inline void synchronizeIfEnable() {
  static const char* mode = std::getenv("DIPU_SYNC_EXEC_MODE");
  if (mode != nullptr) {
    // TODO(log) - use a logger library to do this.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    DIPU_LOG_ONCE << "The synchronous operation is performed after "
                  << "the diopi function call because the DIPU_SYNC_EXEC_MODE "
                     "environment variable is set\n";
    dipu::getCurrentDIPUStream().synchronize();
  }
}

inline bool dipuKeepTorchopDefaultImpl(const char* opname) {
  static const char* env = std::getenv("DIPU_KEEP_TORCHOP_DEFAULT_IMPL_OPS");
  return (env != nullptr) &&
         ((std::string(env) + ',').find(std::string(opname) + ',') <
          (strlen(env) - 1));
}

inline int dumpOpArgLevel() {
  static const char* env_ptr = std::getenv("DIPU_DUMP_OP_ARGS");
  static int level = env_ptr ? std::atoi(env_ptr) : 0;
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
           << ", data_ptr: " << tensor.data_ptr()
           << ", storage_data_ptr: " << tensor.storage().data_ptr().get()
           << ", storage_offset: " << tensor.storage_offset();
    if (dumpOpArgLevel() > 2) {
      stream << '\n' << tensor_clone_to_host(tensor);
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
c10::DimVector infer_reduce_op_shape(const container1<T1>& input_shape,
                                     const container2<T2>& dims, bool keepdim) {
  if (dims.size() <= 0) {
    if (keepdim) {
      return c10::DimVector(input_shape.size(), 1);
    }
    return {};
  }
  if (keepdim) {
    c10::DimVector output_shape(input_shape.begin(), input_shape.end());
    for (auto iter = dims.begin(); iter != dims.end(); ++iter) {
      auto dim = *iter;
      dim += dim < 0 ? input_shape.size() : 0;
      output_shape[dim] = 1;
    }
    return output_shape;
  }
  c10::DimVector output_shape;
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

template <typename T>
decltype(auto) unwrap_or(T&& x, ...) noexcept {
  return std::forward<T>(x);
}

template <typename T, typename U>
auto unwrap_or(T&& x, U&& fallback)
    -> decltype(std::forward<T>(x).value_or(std::forward<U>(fallback))) {
  return std::forward<T>(x).value_or(std::forward<U>(fallback));
}

template <typename... T>
bool is_mixed_type(const T&... tensors) {
  auto is_mixed = at::native::is_mixed_type(tensors...);
  if (is_mixed) {
    at::native::check_mixed_data_type(tensors...);
  }
  return is_mixed;
}

template <typename... Args>
at::ScalarType mixed_output_scalar_type(const at::Tensor& input,
                                        const Args&... parameters) {
  auto static const empty = at::Tensor{};
  auto mixed = is_mixed_type(input, unwrap_or(parameters, empty)...);
  return at::native::param_scalar_type(input, mixed);
}

inline bool is_scalar_on_cpu(const at::Tensor& t) {
  return t.defined() && t.is_cpu() && t.numel() == 1;
}

// This function is used to check if tensor is a scalar tensor by any means.
inline bool is_scalar_tensor(const c10::optional<at::Tensor>& t) {
  return t.has_value() && ((*t).unsafeGetTensorImpl()->is_wrapped_number() ||
                           ((*t).is_cpu() && (*t).numel() == 1));
}

inline bool ignore_device_check(const c10::optional<at::Tensor>& t) {
  return (kDipuVendorDeviceType == devapis::VendorDeviceType::CUDA ||
          kDipuVendorDeviceType == devapis::VendorDeviceType::MUXI) &&
         is_scalar_tensor(t);
}
}  // namespace native
}  // namespace dipu
