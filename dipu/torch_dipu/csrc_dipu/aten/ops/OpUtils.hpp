#pragma once

#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/Log.h>

namespace dipu::native {

inline bool checkDiopiReturnValue() {
  static bool enable =
      std::getenv("DIPU_DISABLE_CHECK_DIOPI_RETURN_VALUE") == nullptr;
  return enable;
}

inline bool checkTensorDevice() {
  static bool enable = []() {
    const char* env_ptr = std::getenv("DIPU_CHECK_TENSOR_DEVICE");
    if (env_ptr == nullptr) {
      return false;
    }
    return std::atoi(env_ptr) > 0 ? true : false;
  }();
  return enable;
}

inline void synchronizeIfEnable() {
  static const char* mode = std::getenv("DIPU_SYNC_EXEC_MODE");
  if (mode != nullptr) {
    DIPU_LOG_ONCE << "The synchronous operation is performed after "
                  << "the diopi function call because the DIPU_SYNC_EXEC_MODE "
                     "environment variable is set"
                  << std::endl;
    dipu::getCurrentDIPUStream().synchronize();
  }
  return;
}

inline int dumpOpArgLevel() {
  static const char* env_ptr = std::getenv("DIPU_DUMP_OP_ARGS");
  static int level = env_ptr ? std::atoi(env_ptr) : 0;
  return level;
}

template <typename T>
static std::string dumpArg(const T& t) {
  std::stringstream stream;
  stream << t;
  return stream.str();
}

template <typename T1>
static std::string dumpArg(const c10::optional<T1>& opt_t) {
  std::stringstream stream;
  if (opt_t.has_value()) {
    stream << dumpArg(opt_t.value());
  }
  return stream.str();
}

template <typename T>
static std::string dumpArg(const c10::OptionalArrayRef<T>& opt_t) {
  std::stringstream stream;
  if (opt_t.has_value()) {
    stream << dumpArg(opt_t.value());
  }
  return stream.str();
}

template <typename T1, template <typename elem> class container>
static std::string dumpArg(const container<T1>& t) {
  std::stringstream stream;
  for (auto iter = t.begin(); iter != t.end(); ++iter) {
    stream << dumpArg(*iter) << ", ";
  }
  return stream.str();
}

template <>
std::string dumpArg(const at::Tensor& tensor) {
  std::stringstream stream;
  if (tensor.defined()) {
    stream << "numel: " << tensor.numel() << ",sizes: " << tensor.sizes()
           << ", stride: " << tensor.strides()
           << ", is_view: " << tensor.is_view() << ", dtype: " << tensor.dtype()
           << ", device:" << tensor.device() << ", layout:" << tensor.layout()
           << ", requires_grad: " << (tensor.requires_grad() ? "true" : "false")
           << ", pinned_memory: " << (tensor.is_pinned() ? "true" : "false")
           << ", memory_format: " << tensor.suggest_memory_format()
           << ",  data_ptr: " << tensor.data_ptr();
    if (dumpOpArgLevel() > 2) {
      stream << std::endl << tensor;
    }
  } else {
    stream << "undefined";
  }
  return stream.str();
}

template <>
std::string dumpArg(const at::Scalar& scalar) {
  std::stringstream stream;
  stream << scalar;
  return stream.str();
}

template <>
std::string dumpArg(const c10::string_view& str) {
  return dumpArg(std::string(str.data()));
}

template <>
std::string dumpArg(const at::Generator& generator) {
  return "";
}

template <typename T, size_t N>
static std::string dumpArg(const std::array<T, N>& t) {
  std::stringstream stream;
  for (auto iter = t.begin(); iter != t.end(); ++iter) {
    stream << dumpArg(*iter) << " ";
  }
  return stream.str();
}

template <>
std::string dumpArg(const c10::List<c10::optional<at::Tensor>>& t) {
  std::stringstream stream;
  stream << "size:" << t.size() << std::endl;
  for (int i = 0; i < t.size(); ++i) {
    bool has_value = t[i].has_value();
    stream << "\t" << i << "th: has_value:" << has_value << " ";
    if (has_value) {
      stream << dumpArg(t[i].value());
    }
    stream << std::endl;
  }
  return stream.str();
}

template <typename T1, typename T2, template <typename elem1> class container1,
          template <typename elem2> class container2>
static std::vector<int64_t> infer_reduce_op_shape(
    const container1<T1>& input_shape, const container2<T2>& dims,
    bool keepdim) {
  if (dims.size() <= 0) {
    return std::vector<int64_t>();
  }
  if (keepdim) {
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    for (auto iter = dims.begin(); iter != dims.end(); ++iter) {
      auto dim = *iter;
      dim += dim < 0 ? input_shape.size() : 0;
      output_shape[dim] = 1;
    }
    return output_shape;
  } else {
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
      if (reduce_dim == false) {
        output_shape.push_back(input_shape.at(i));
      }
    }
    return output_shape;
  }
}

static std::string _allclose(const at::Tensor& a, const at::Tensor& b) {
  if (a.defined() && b.defined()) {
    try {
      if (at::allclose(a.cpu(), b.cpu(), 1e-4, 1e-5, true)) {
        return "allclose";
      } else {
        auto diff = at::abs(a.cpu() - b.cpu());
        auto mae = diff.mean().item<double>();
        auto max_diff = diff.max().item<double>();
        return "not_close, max diff: " + std::to_string(max_diff) +
               ", MAE: " + std::to_string(mae);
      }
    } catch (...) {
      return "compare_error: not_close";
    }
  } else {
    if (a.defined() != b.defined()) {
      return "not_close, one of tensor inputs is empty";
    } else {
      return "allclose";
    }
  }
}

static std::string _allclose(const c10::ArrayRef<at::Tensor>& a,
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

}  // namespace dipu::native
