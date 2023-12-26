// Copyright (c) 2023, DeepLink.

#include "DIPUAmp.hpp"

#include "ATen/Operators.h"
#include "ATen/autocast_mode.h"
#include "torch/library.h"

#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/device/basedef.h"

#ifndef DIPU_NO_VENDOR_AUTOCAST
#include "csrc_dipu/vendor/vendor_autocast.h"
#endif

namespace at {
namespace autocast {

// NOLINTBEGIN
// ----------------------------------------------------------------------------
// Code from aten/src/ATen/autocast_mode.cpp
// SHOULD BE REMOVED AFTER PYTORCH 2.1
// since they have been moved into autocast_mode.h
// ----------------------------------------------------------------------------
namespace {

template <typename... Args>
inline bool firstarg_is_eligible(DeviceType device_type, const Tensor& arg,
                                 Args... args) {
  return is_eligible(arg, device_type);
}

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  lower_precision_fp =
      0,  // Cast all inputs to lower_precision_fp before running the op.
          // Currently, lower_precision_fp is fp16 for AutocastCUDA, and is
          // defined by user(default bf16) for AutocastCPU.
  fp32,   // Cast all inputs to at::kFloat before running the op.
  fp32_set_opt_dtype,  // Treats functions (like softmax) that
                       //   1. we'd like to run in fp32 and
                       //   2. have a c10::optional<ScalarType> arg that
                       //   controls the output type.
                       // fp32_set_opt_dtype wrappers' policy is:  if the output
                       // type is already set, don't touch it, otherwise, set it
                       // to at::kFloat.
  fp32_append_dtype,   // Treats functions (like norm) that
                       //   1. we'd like to run in fp32 and
                       //   2. have some overloads that accept an output type
                       //   and other overloads that don't.
                       // fp32_append_dtype wrappers wrap the overloads that
                       // don't have an output dtype. The wrapper policy is:
                       // append at::kFloat to the args, and redispatch to the
  // type-aware overload.
  promote,  // Run in the widest dtype among several args.
};

// Base template for WrapFunction_, which is specialized to contain a "call"
// method each CastPolicy
template <CastPolicy policy, DeviceType device_type, class Redispatch,
          Redispatch* F, class Ret, class ArgList>
struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch* F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::lower_precision_fp, device_type, Redispatch, F,
                     Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    // DispatchKey::Autocast is not the alias key of all AutocastType as
    // Autograd, it's just alias of AutocastCUDA (see c10/core/DispatchKey.h)
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(
        cached_cast(get_lower_precision_fp_from_device_type(device_type), args,
                    device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch* F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::fp32, device_type, Redispatch, F, Ret,
                     guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(at::kFloat, args, device_type)...);
  }
};

// CastPolicy::fp32_set_opt_dtype General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch* F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::fp32_set_opt_dtype, device_type, Redispatch, F,
                     Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    if (firstarg_is_eligible(device_type, args...)) {
      return (*F)(set_opt_dtype(at::kFloat, args)...);
    } else {
      // If ineligible, calls F with unaltered args.  Does not set opt dtype,
      // because setting opt dtype explicitly may interfere with internal
      // implicit promotion decisions.
      return (*F)(args...);
    }
  }
};

// CastPolicy::fp32_append_dtype General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch* F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::fp32_append_dtype, device_type, Redispatch, F,
                     Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
#ifdef DIPU_TORCH200
    at::ScalarType out_type = type_from_firstarg(at::kFloat, args...);
#else  // # DIPU_TORCH201 or higher
    at::ScalarType out_type =
        type_from_firstarg(device_type, at::kFloat, args...);
#endif
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch* F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::promote, device_type, Redispatch, F, Ret,
                     guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    auto to_type =
        promote_type(get_lower_precision_fp_from_device_type(device_type),
                     device_type, args...);
    return (*F)(cached_cast(to_type, args, device_type)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating
// core/boxing/impl/WrapFunctionIntoFunctor.h)
template <
    CastPolicy policy, DeviceType device_type,
    class Registered,  // The signature for which we're registering.  The
                       // dispatcher's calling code invokes our registered
                       // functions with arguments matching Registered, so we
                       // register WrapFunction_::call methods with a matching
                       // signature to properly field those arguments.
    // guts::function_traits below extracts return_type and
    // parameter_types from Registered, which WrapFunction_
    // templates above use to declare their call methods.
    class Redispatch,  // The signature for the function we're redispatching to.
                       // In most cases this is the same as Registered, but for
                       // some ops (for example, ops where we append a dtype)
                       // it's useful to redispatch to a function with a
                       // different signature.
    Redispatch* F>     // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<
      policy, device_type, Redispatch, F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

}  // namespace
// ---------- CODE THAT NEEDS TO BE REMOVED AFTER PYTORCH 2.1 ENDED -----------

// ----------------------------------------------------------------------------
// THESE MACROS SHOULD BE REFACTORED AFTER PYTORCH 2.1
// ----------------------------------------------------------------------------

// KERNEL_DIPU registration for AutocastDIPU
#define KERNEL_DIPU(OP, POLICY)                                      \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP),                         \
         &WrapFunction<CastPolicy::POLICY, dipu::DIPU_DEVICE_TYPE,   \
                       decltype(ATEN_FN(OP)), decltype(ATEN_FN(OP)), \
                       &ATEN_FN(OP)>::type::call);
#define KERNEL_DIPU2(OP, OVERLOAD, POLICY)                         \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD),         \
         &WrapFunction<CastPolicy::POLICY, dipu::DIPU_DEVICE_TYPE, \
                       decltype(ATEN_FN2(OP, OVERLOAD)),           \
                       decltype(ATEN_FN2(OP, OVERLOAD)),           \
                       &ATEN_FN2(OP, OVERLOAD)>::type::call);

// Less-common but still useful case: redispatching to a function with a new
// signature (e.g. appending a dtype)
#define KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(                           \
    REDISPATCH_FUNC, REGISTER_NAME, REGISTER_SIGNATURE, REDISPATCH_SIGNATURE, \
    POLICY)                                                                   \
  m.impl(TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME),                        \
         &WrapFunction<CastPolicy::POLICY, dipu::DIPU_DEVICE_TYPE,            \
                       REGISTER_SIGNATURE, REDISPATCH_SIGNATURE,              \
                       &REDISPATCH_FUNC>::type::call);
// ---------------------------  MACROS ENDED  ---------------------------------
// NOLINTEND

namespace {

// Compile-time type conversion from dupu::autocast::CastPolicy
template <dipu::autocast::DipuCastPolicy>
struct FromDipuCastPolicyHelper {
  static void MaybeLogError() {
    DIPU_LOGE("invalid cast policy, fallback to fp32.");
  }
  static constexpr CastPolicy kPolicy = CastPolicy::fp32;
};

#define DIPU_DEFINE_CAST_POLICY_CONVERSION(DIPU_POLICY, POLICY) \
  template <>                                                   \
  struct FromDipuCastPolicyHelper<                              \
      dipu::autocast::DipuCastPolicy::DIPU_POLICY> {            \
    static void MaybeLogError() {}                              \
    static constexpr CastPolicy kPolicy = CastPolicy::POLICY;   \
  };

DIPU_DEFINE_CAST_POLICY_CONVERSION(kLowerPrecisionFp, lower_precision_fp);
DIPU_DEFINE_CAST_POLICY_CONVERSION(kFp32, fp32);
DIPU_DEFINE_CAST_POLICY_CONVERSION(kPromote, promote);

#undef DIPU_DEFINE_CAST_POLICY_CONVERSION

// Bind macros with customizable cast policies.
// Modified from KERNEL_DIPU/KERNEL_DIPU2
#define DIPU_AUTOCAST_BIND(OP)                                           \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP),                             \
         &WrapFunction<                                                  \
             FromDipuCastPolicyHelper<DIPU_OP_CAST_POLICY(OP)>::kPolicy, \
             dipu::DIPU_DEVICE_TYPE, decltype(ATEN_FN(OP)),              \
             decltype(ATEN_FN(OP)), &ATEN_FN(OP)>::type::call);
#define DIPU_AUTOCAST_BIND2(OP, OVERLOAD)                                \
  m.impl(TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD),               \
         &WrapFunction<                                                  \
             FromDipuCastPolicyHelper<DIPU_OP_CAST_POLICY(OP)>::kPolicy, \
             dipu::DIPU_DEVICE_TYPE, decltype(ATEN_FN2(OP, OVERLOAD)),   \
             decltype(ATEN_FN2(OP, OVERLOAD)),                           \
             &ATEN_FN2(OP, OVERLOAD)>::type::call);

// This function will throw an error message when
// torch.nn.functional.binary_cross_entropy is called within an autocast block
Tensor DipuBinaryCrossEntropyBanned(const Tensor&, const Tensor&,
                                    const c10::optional<Tensor>&, int64_t) {
  AT_ERROR(
      R"(torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
Many models use a sigmoid layer right before the binary cross entropy layer.
In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits
or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are
safe to autocast.)");
}

TORCH_LIBRARY_IMPL(_, DIPU_AUTOCAST_DEVICE_TYPE_MACRO, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, DIPU_AUTOCAST_DEVICE_TYPE_MACRO, m) {
  // lower_precision_fp
  DIPU_AUTOCAST_BIND2(_convolution, deprecated)
  DIPU_AUTOCAST_BIND(_convolution)
  DIPU_AUTOCAST_BIND(conv1d)
  DIPU_AUTOCAST_BIND(conv2d)
  DIPU_AUTOCAST_BIND(conv3d)
  DIPU_AUTOCAST_BIND(conv_tbc)
  DIPU_AUTOCAST_BIND(conv_transpose1d)
  DIPU_AUTOCAST_BIND2(conv_transpose2d, input)
  DIPU_AUTOCAST_BIND2(conv_transpose3d, input)
  DIPU_AUTOCAST_BIND(convolution)
  DIPU_AUTOCAST_BIND(cudnn_convolution)
  DIPU_AUTOCAST_BIND(cudnn_convolution_transpose)
  DIPU_AUTOCAST_BIND(prelu)
  DIPU_AUTOCAST_BIND(addmm)
  DIPU_AUTOCAST_BIND(addmv)
  DIPU_AUTOCAST_BIND(addr)
  DIPU_AUTOCAST_BIND(matmul)
  DIPU_AUTOCAST_BIND(einsum)
  DIPU_AUTOCAST_BIND(mm)
  DIPU_AUTOCAST_BIND(mv)
  DIPU_AUTOCAST_BIND(linear)
  DIPU_AUTOCAST_BIND(addbmm)
  DIPU_AUTOCAST_BIND(baddbmm)
  DIPU_AUTOCAST_BIND(bmm)
  DIPU_AUTOCAST_BIND(chain_matmul)
  DIPU_AUTOCAST_BIND(linalg_multi_dot)
  DIPU_AUTOCAST_BIND(_thnn_fused_lstm_cell)
  DIPU_AUTOCAST_BIND(_thnn_fused_gru_cell)
  DIPU_AUTOCAST_BIND(lstm_cell)
  DIPU_AUTOCAST_BIND(gru_cell)
  DIPU_AUTOCAST_BIND(rnn_tanh_cell)
  DIPU_AUTOCAST_BIND(rnn_relu_cell)
  DIPU_AUTOCAST_BIND(_scaled_dot_product_flash_attention)
  DIPU_AUTOCAST_BIND(scaled_dot_product_attention)

  // fp32
  DIPU_AUTOCAST_BIND(acos)
  DIPU_AUTOCAST_BIND(asin)
  DIPU_AUTOCAST_BIND(cosh)
  DIPU_AUTOCAST_BIND(erfinv)
  DIPU_AUTOCAST_BIND(exp)
  DIPU_AUTOCAST_BIND(expm1)
  DIPU_AUTOCAST_BIND(log)
  DIPU_AUTOCAST_BIND(log10)
  DIPU_AUTOCAST_BIND(log2)
  DIPU_AUTOCAST_BIND(log1p)
  DIPU_AUTOCAST_BIND(reciprocal)
  DIPU_AUTOCAST_BIND(rsqrt)
  DIPU_AUTOCAST_BIND(sinh)
  DIPU_AUTOCAST_BIND(tan)
  DIPU_AUTOCAST_BIND2(pow, Tensor_Scalar)
  DIPU_AUTOCAST_BIND2(pow, Tensor_Tensor)
  DIPU_AUTOCAST_BIND2(pow, Scalar)
  DIPU_AUTOCAST_BIND(softplus)
  DIPU_AUTOCAST_BIND(layer_norm)
  DIPU_AUTOCAST_BIND(native_layer_norm)
  DIPU_AUTOCAST_BIND(group_norm)
  DIPU_AUTOCAST_BIND2(frobenius_norm, dim)
  DIPU_AUTOCAST_BIND(nuclear_norm)
  DIPU_AUTOCAST_BIND2(nuclear_norm, dim)
  DIPU_AUTOCAST_BIND(cosine_similarity)
  DIPU_AUTOCAST_BIND(poisson_nll_loss)
  DIPU_AUTOCAST_BIND(cosine_embedding_loss)
  DIPU_AUTOCAST_BIND(nll_loss)
  DIPU_AUTOCAST_BIND(nll_loss2d)
  DIPU_AUTOCAST_BIND(hinge_embedding_loss)
  DIPU_AUTOCAST_BIND(kl_div)
  DIPU_AUTOCAST_BIND(l1_loss)
  DIPU_AUTOCAST_BIND(smooth_l1_loss)
  DIPU_AUTOCAST_BIND(huber_loss)
  DIPU_AUTOCAST_BIND(mse_loss)
  DIPU_AUTOCAST_BIND(margin_ranking_loss)
  DIPU_AUTOCAST_BIND(multilabel_margin_loss)
  DIPU_AUTOCAST_BIND(soft_margin_loss)
  DIPU_AUTOCAST_BIND(triplet_margin_loss)
  DIPU_AUTOCAST_BIND(multi_margin_loss)
  DIPU_AUTOCAST_BIND(binary_cross_entropy_with_logits)
  DIPU_AUTOCAST_BIND(dist)
  DIPU_AUTOCAST_BIND(pdist)
  DIPU_AUTOCAST_BIND(cdist)
  DIPU_AUTOCAST_BIND(renorm)
  DIPU_AUTOCAST_BIND(logsumexp)
  // fp32_set_opt_dtype
  KERNEL_DIPU(prod, fp32_set_opt_dtype)
  KERNEL_DIPU2(prod, dim_int, fp32_set_opt_dtype)
  KERNEL_DIPU2(prod, dim_Dimname, fp32_set_opt_dtype)
  KERNEL_DIPU2(softmax, int, fp32_set_opt_dtype)
  KERNEL_DIPU2(softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_DIPU2(log_softmax, int, fp32_set_opt_dtype)
  KERNEL_DIPU2(log_softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_DIPU(cumprod, fp32_set_opt_dtype)
  KERNEL_DIPU2(cumprod, dimname, fp32_set_opt_dtype)
  KERNEL_DIPU(cumsum, fp32_set_opt_dtype)
  KERNEL_DIPU2(cumsum, dimname, fp32_set_opt_dtype)
  KERNEL_DIPU(linalg_vector_norm, fp32_set_opt_dtype)
  KERNEL_DIPU(linalg_matrix_norm, fp32_set_opt_dtype)
  KERNEL_DIPU2(linalg_matrix_norm, str_ord, fp32_set_opt_dtype)
  // commenting these out because they accept an explicit (not-optional) dtype,
  // and we shouldn't try to flip that even when autocasting. KERNEL_DIPU2(norm,
  // ScalarOpt_dtype, fp32_set_opt_dtype) KERNEL_DIPU2(norm,
  // ScalarOpt_dim_dtype, fp32_set_opt_dtype) KERNEL_DIPU2(norm,
  // names_ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  KERNEL_DIPU(sum, fp32_set_opt_dtype)
  KERNEL_DIPU2(sum, dim_IntList, fp32_set_opt_dtype)
  KERNEL_DIPU2(sum, dim_DimnameList, fp32_set_opt_dtype)
  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this
  // policy.
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(
      at::norm, "norm.Scalar", Tensor(const Tensor&, const Scalar&),
      Tensor(const Tensor&, const c10::optional<Scalar>&, ScalarType),
      fp32_append_dtype)
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(
      at::norm, "norm.ScalarOpt_dim",
      Tensor(const Tensor&, const c10::optional<Scalar>&, IntArrayRef, bool),
      Tensor(const Tensor&, const c10::optional<Scalar>&, IntArrayRef, bool,
             ScalarType),
      fp32_append_dtype)
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(
      at::norm, "norm.names_ScalarOpt_dim",
      Tensor(const Tensor&, const c10::optional<Scalar>&, DimnameList, bool),
      Tensor(const Tensor&, const c10::optional<Scalar>&, DimnameList, bool,
             ScalarType),
      fp32_append_dtype)
  // promote
  DIPU_AUTOCAST_BIND(addcdiv)
  DIPU_AUTOCAST_BIND(addcmul)
  DIPU_AUTOCAST_BIND(atan2)
  DIPU_AUTOCAST_BIND(bilinear)
  DIPU_AUTOCAST_BIND(cross)
  DIPU_AUTOCAST_BIND(dot)
  DIPU_AUTOCAST_BIND(grid_sampler)
  DIPU_AUTOCAST_BIND(index_put)
  DIPU_AUTOCAST_BIND(tensordot)
  DIPU_AUTOCAST_BIND(scatter_add)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::DipuBinaryCrossEntropyBanned)));
}

}  // namespace

}  // namespace autocast
}  // namespace at
