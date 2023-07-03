# Copyright (c) 2023, DeepLink.
diopi_wrapper_file_template_content = \
"""
// autogened file
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <type_traits>

#include <torch/csrc/autograd/custom_function.h>
#include <torch/types.h>
#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/aten/RegisterDIPU.hpp"
#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "CustomFallbackFunctions.hpp"

$header_include_code

namespace dipu::native {

inline bool checkDiopiReturnValue() {
    static bool enable = std::getenv("DIPU_DISABLE_CHECK_DIOPI_RETURN_VALUE") == nullptr;
    return enable;
}

inline void synchronizeIfEnable() {
    static const char* mode = std::getenv("DIPU_SYNC_EXEC_MODE");
    if (mode != nullptr) {
        DIPU_LOG_ONCE << "The synchronous operation is performed after "
            <<"the diopi function call because the DIPU_SYNC_EXEC_MODE environment variable is set" << std::endl;
        dipu::getCurrentDIPUStream().synchronize();
    }
    return;
}

inline int dumpOpArgLevel() {
    const char* env_ptr = std::getenv("DIPU_DUMP_OP_ARGS");
    int level = env_ptr ? std::atoi(env_ptr) : 0;
    return level;
}

template<typename T>
static std::string dumpArg(const T& t) {
    std::stringstream stream;
    stream << t;
    return stream.str();
}

template<typename T1>
static std::string dumpArg(const c10::optional<T1>  & opt_t) {
    std::stringstream stream;
    if (opt_t.has_value()) {
        stream << dumpArg(opt_t.value());
    }
    return stream.str();
}

template<typename T>
static std::string dumpArg(const c10::OptionalArrayRef<T>  & opt_t) {
    std::stringstream stream;
    if (opt_t.has_value()) {
        stream << dumpArg(opt_t.value());
    }
    return stream.str();
}

template<typename T1, template<typename elem> class container>
static std::string dumpArg(const container<T1> & t) {
    std::stringstream stream;
    for (auto iter = t.begin(); iter != t.end(); ++iter) {
        stream << dumpArg(*iter) << ", ";
    }
    return stream.str();
}

template<>
std::string dumpArg(const at::Tensor& tensor) {
    std::stringstream stream;
    if (tensor.defined()) {
        stream << "numel:" << tensor.numel() << ",sizes:" << tensor.sizes() << ", stride:" << tensor.strides() << ",is_view:" << tensor.is_view() << "," << "dtype=" << tensor.dtype()
        << ", device=" << tensor.device() << ", layout=" << tensor.layout() << ", requires_grad=" << (tensor.requires_grad() ? "true" : "false") << ", pinned_memory=" << (tensor.is_pinned() ? "true" : "false") 
        << ", memory_format="  << tensor.suggest_memory_format() << ", data_ptr:" << tensor.data_ptr();
        if (dumpOpArgLevel() > 2) {
            stream << std::endl << tensor;
        }
    } else {
        stream << "undefined";
    }
    return stream.str();
}

template<>
std::string dumpArg(const at::Scalar& scalar) {
    std::stringstream stream;
    stream << scalar;
    return stream.str();
}

template<>
std::string dumpArg(const c10::string_view& str) {
    return dumpArg(std::string(str.data()));
}

template<>
std::string dumpArg(const at::Generator& generator) {
    return "";
}

template<typename T, size_t N>
static std::string dumpArg(const std::array<T, N>& t) {
    std::stringstream stream;
    for (auto iter = t.begin(); iter != t.end(); ++iter) {
        stream << dumpArg(*iter) << " ";
    }
    return stream.str();
}

template<>
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


template<typename T1, typename T2 , template<typename elem1> class container1, template<typename elem2> class container2>
static std::vector<int64_t> infer_reduce_op_shape(const container1<T1> & input_shape, const container2<T2> & dims, bool keepdim) {
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
    if(a.defined() && b.defined()) {
        try {
            return at::allclose(a.cpu(), b.cpu(), 1e-3, 1e-3, true) ? "allclose" : "not_close";
        } catch (...) {
            return "compare_fail: not_close";
        }
    } else {
        if(a.defined() != b.defined()) {
            return "not_close";
        } else {
            return "allclose";
        }
    }
}

using namespace dipu::diopi_helper;

$functions_code


}  // namespace dipu::native

namespace at {

TORCH_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
    $op_register_code
}

TORCH_LIBRARY_IMPL(aten, DIPU_AUTOGRAD_DEVICE_TYPE_MACRO, m) {
    $autograd_op_register_code
}

}  // namespace at

"""

diopi_wrapper_function_template_content = \
"""
//  $comment
$cppsignautre {
    ::diopiContext context(dipu::getCurrentDIPUStream().rawstream());
    auto ctx = &context;

    $custom_code_at_the_beginning

    $input_process_code

    $output_process_code

    $attrs_process_code

    $custom_code_before_call_diopi

    ::diopiError_t ret = $diopi_fun_call_code
    if (checkDiopiReturnValue()) {
        TORCH_CHECK(ret == ::diopiSuccess, __FILE__, ":", __LINE__, R"($diopi_fun_call_code)", " error, error code is ", ret, "error message is ", diopiGetLastErrorString());
    }

    $custom_code_before_return

    synchronizeIfEnable();

    $return_code
}
"""

op_register_template_content = \
"""
DIOPI_ATEN_FUNC("$register_name", $diopi_fun_name, $aten_fun_name);
"""

op_with_custom_fallback_register_template_content = \
"""
DIOPI_ATEN_FUNC_CUSTOM_FALLBACK("$register_name", $diopi_fun_name, $force_fallback /*whether force fallback*/, $aten_fun_name, $fallbackFunc);
"""

custom_autograd_template_content = \
"""
class $autograd_function_name : public torch::autograd::Function<$autograd_function_name> {
public:
    static $return_code forward(torch::autograd::AutogradContext *ctx, $param_list) {
        $forward_process_code

        $save_for_backward_code

        at::AutoDispatchBelowADInplaceOrView g;
        return $call_forward_impl_code;
    }

  static std::vector<at::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<at::Tensor> grad_outputs) {
      $load_saved_data_code

      $cal_grad_code

      $call_backward_impl_code

      $backward_return_code
  }
};

$cppsignautre {
    auto result = $autograd_function_name::apply($arg_name_list);
    $wrappter_custom_return
}
"""


autocompare_template_content = \
"""
//  $comment
$cppsignautre {
    std::cout << std::endl << __FUNCTION__ << std::endl;
    $transform_input_to_cpu_code

    $execute_op_on_cpu_code

    $execute_op_on_device_code

    $transform_result_to_cpu_code

    $result_compare_code
}
"""