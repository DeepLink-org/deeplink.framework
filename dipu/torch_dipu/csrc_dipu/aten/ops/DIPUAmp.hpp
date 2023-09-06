#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/NativeFunctions.h>
#include <ATen/autocast_mode.h>
#include <ATen/Operators.h>

#include <c10/util/intrusive_ptr.h>
#include <c10/core/impl/LocalDispatchKeySet.h>

#include <csrc_dipu/base/basedef.h>

#include <iostream>
#include <exception>
#include <mutex>

// Most of code from aten/src/ATen/autocast_mode.cpp

namespace at {
namespace autocast {

// KERNEL_DIPU registration for AutocastDIPU
#define KERNEL_DIPU(OP, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP), \
    &WrapFunction<CastPolicy::POLICY, DeviceTypeDIPU, decltype(ATEN_FN(OP)), decltype(ATEN_FN(OP)), &ATEN_FN(OP)>::type::call);
#define KERNEL_DIPU2(OP, OVERLOAD, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
    &WrapFunction<CastPolicy::POLICY, DeviceTypeDIPU, decltype(ATEN_FN2(OP, OVERLOAD)), decltype(ATEN_FN2(OP, OVERLOAD)), &ATEN_FN2(OP, OVERLOAD)>::type::call);

// Less-common but still useful case: redispatching to a function with a new signature (e.g. appending a dtype)
#define KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(REDISPATCH_FUNC, REGISTER_NAME, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, POLICY) \
  m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
    &WrapFunction<CastPolicy::POLICY, DeviceTypeDIPU, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, &REDISPATCH_FUNC>::type::call);


Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type);

/*******************************
Banned functions
*******************************/
// Origin function is invisible, so we rewrite it.
Tensor dipu_binary_cross_entropy_banned(const Tensor &, const Tensor &, const c10::optional<Tensor>&, int64_t) {
  AT_ERROR("torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
           "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
           "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
           "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
           "safe to autocast.");
}


// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  lower_precision_fp = 0, // Cast all inputs to lower_precision_fp before running the op.
                          // Currently, lower_precision_fp is fp16 for AutocastCUDA, and is defined by user(default bf16) for AutocastCPU.
  fp32, // Cast all inputs to at::kFloat before running the op.
  fp32_set_opt_dtype, // Treats functions (like softmax) that
                      //   1. we'd like to run in fp32 and
                      //   2. have a c10::optional<ScalarType> arg that controls the output type.
                      // fp32_set_opt_dtype wrappers' policy is:  if the output type is already set,
                      // don't touch it, otherwise, set it to at::kFloat.
  fp32_append_dtype, // Treats functions (like norm) that
                     //   1. we'd like to run in fp32 and
                     //   2. have some overloads that accept an output type and other overloads that don't.
                     // fp32_append_dtype wrappers wrap the overloads that don't have an output dtype.
                     // The wrapper policy is:  append at::kFloat to the args, and redispatch to the
                     // type-aware overload.
  promote, // Run in the widest dtype among several args.
};

// Base template for WrapFunction_, which is specialized to contain a "call" method each CastPolicy
template<CastPolicy policy, DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class ArgList> struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::lower_precision_fp, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(get_lower_precision_fp_from_device_type(device_type), args, device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(at::kFloat, args, device_type)...);
  }
};

// CastPolicy::fp32_set_opt_dtype General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32_set_opt_dtype, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);
    if (firstarg_is_eligible(args...)) {
      return (*F)(set_opt_dtype(at::kFloat, args)...);
    } else {
      // If ineligible, calls F with unaltered args.  Does not set opt dtype, because setting
      // opt dtype explicitly may interfere with internal implicit promotion decisions.
      return (*F)(args...);
    }
  }
};

// CastPolicy::fp32_append_dtype General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::fp32_append_dtype, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);
    at::ScalarType out_type = type_from_firstarg(at::kFloat, args...);
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote General_DeviceType
template<DeviceType device_type, class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<CastPolicy::promote, device_type, Redispatch, F, Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(get_autocast_dispatch_key_from_device_type(device_type));
    auto to_type = promote_type(get_lower_precision_fp_from_device_type(device_type), device_type, args...);
    return (*F)(cached_cast(to_type, args, device_type)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating core/boxing/impl/WrapFunctionIntoFunctor.h)
template<CastPolicy policy,
         DeviceType device_type,
         class Registered, // The signature for which we're registering.  The dispatcher's calling code invokes our
                           // registered functions with arguments matching Registered, so we register
                           // WrapFunction_::call methods with a matching signature to properly field those arguments.
                           // guts::function_traits below extracts return_type and parameter_types from Registered,
                           // which WrapFunction_ templates above use to declare their call methods.
         class Redispatch, // The signature for the function we're redispatching to.  In most cases this is the same
                           // as Registered, but for some ops (for example, ops where we append a dtype) it's useful
                           // to redispatch to a function with a different signature.
         Redispatch* F>    // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<policy,
                             device_type,
                             Redispatch,
                             F,
                             typename guts::function_traits<Registered>::return_type,
                             typename guts::function_traits<Registered>::parameter_types>;
};

} // namespace autocast
} // namespace at
