#include "DIPUAmp.hpp"

#include <ATen/Operators.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

#include <csrc_dipu/base/basedef.h>

namespace at {
namespace autocast {

// NOLINTBEGIN
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

// ----------------------------------------------------------------------------
// Code from aten/src/ATen/autocast_mode.cpp
// SHOULD BE REMOVED AFTER PYTORCH 2.1
// since they have been moved into autocast_mode.h
// ----------------------------------------------------------------------------
namespace {

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
          Redispatch *F, class Ret, class ArgList>
struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch *F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::lower_precision_fp, device_type, Redispatch, F,
                     Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(
        cached_cast(get_lower_precision_fp_from_device_type(device_type), args,
                    device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch *F, class Ret,
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
template <DeviceType device_type, class Redispatch, Redispatch *F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::fp32_set_opt_dtype, device_type, Redispatch, F,
                     Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(DispatchKey::Autocast);
    if (firstarg_is_eligible(args...)) {
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
template <DeviceType device_type, class Redispatch, Redispatch *F, class Ret,
          class... Args>
struct WrapFunction_<CastPolicy::fp32_append_dtype, device_type, Redispatch, F,
                     Ret, guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    at::ScalarType out_type = type_from_firstarg(at::kFloat, args...);
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote General_DeviceType
template <DeviceType device_type, class Redispatch, Redispatch *F, class Ret,
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
    Redispatch *F>     // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<
      policy, device_type, Redispatch, F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

}  // namespace
// ---------- CODE THAT NEEDS TO BE REMOVED AFTER PYTORCH 2.1 ENDED ------------
// NOLINTEND

namespace {

// This function will throw an error message when
// torch.nn.functional.binary_cross_entropy is called within an autocast block
Tensor DipuBinaryCrossEntropyBanned(const Tensor &, const Tensor &,
                                        const c10::optional<Tensor> &,
                                        int64_t) {
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
  KERNEL_DIPU2(_convolution, deprecated, lower_precision_fp)
  KERNEL_DIPU(_convolution, lower_precision_fp)
  KERNEL_DIPU(conv1d, lower_precision_fp)
  KERNEL_DIPU(conv2d, lower_precision_fp)
  KERNEL_DIPU(conv3d, lower_precision_fp)
  KERNEL_DIPU(conv_tbc, lower_precision_fp)
  KERNEL_DIPU(conv_transpose1d, lower_precision_fp)
  KERNEL_DIPU2(conv_transpose2d, input, lower_precision_fp)
  KERNEL_DIPU2(conv_transpose3d, input, lower_precision_fp)
  KERNEL_DIPU(convolution, lower_precision_fp)
  KERNEL_DIPU(cudnn_convolution, lower_precision_fp)
  KERNEL_DIPU(cudnn_convolution_transpose, lower_precision_fp)
  KERNEL_DIPU(prelu, lower_precision_fp)
  KERNEL_DIPU(addmm, lower_precision_fp)
  KERNEL_DIPU(addmv, lower_precision_fp)
  KERNEL_DIPU(addr, lower_precision_fp)
  KERNEL_DIPU(matmul, lower_precision_fp)
  KERNEL_DIPU(einsum, lower_precision_fp)
  KERNEL_DIPU(mm, lower_precision_fp)
  KERNEL_DIPU(mv, lower_precision_fp)
  KERNEL_DIPU(linear, lower_precision_fp)
  KERNEL_DIPU(addbmm, lower_precision_fp)
  KERNEL_DIPU(baddbmm, lower_precision_fp)
  KERNEL_DIPU(bmm, lower_precision_fp)
  KERNEL_DIPU(chain_matmul, lower_precision_fp)
  KERNEL_DIPU(linalg_multi_dot, lower_precision_fp)
  KERNEL_DIPU(_thnn_fused_lstm_cell, lower_precision_fp)
  KERNEL_DIPU(_thnn_fused_gru_cell, lower_precision_fp)
  KERNEL_DIPU(lstm_cell, lower_precision_fp)
  KERNEL_DIPU(gru_cell, lower_precision_fp)
  KERNEL_DIPU(rnn_tanh_cell, lower_precision_fp)
  KERNEL_DIPU(rnn_relu_cell, lower_precision_fp)
  KERNEL_DIPU(_scaled_dot_product_flash_attention, lower_precision_fp)
  KERNEL_DIPU(scaled_dot_product_attention, lower_precision_fp)

  // fp32
  KERNEL_DIPU(acos, fp32)
  KERNEL_DIPU(asin, fp32)
  KERNEL_DIPU(cosh, fp32)
  KERNEL_DIPU(erfinv, fp32)
  KERNEL_DIPU(exp, fp32)
  KERNEL_DIPU(expm1, fp32)
  KERNEL_DIPU(log, fp32)
  KERNEL_DIPU(log10, fp32)
  KERNEL_DIPU(log2, fp32)
  KERNEL_DIPU(log1p, fp32)
  KERNEL_DIPU(reciprocal, fp32)
  KERNEL_DIPU(rsqrt, fp32)
  KERNEL_DIPU(sinh, fp32)
  KERNEL_DIPU(tan, fp32)
  KERNEL_DIPU2(pow, Tensor_Scalar, fp32)
  KERNEL_DIPU2(pow, Tensor_Tensor, fp32)
  KERNEL_DIPU2(pow, Scalar, fp32)
  KERNEL_DIPU(softplus, fp32)
  KERNEL_DIPU(layer_norm, fp32)
  KERNEL_DIPU(native_layer_norm, fp32)
  KERNEL_DIPU(group_norm, fp32)
  KERNEL_DIPU2(frobenius_norm, dim, fp32)
  KERNEL_DIPU(nuclear_norm, fp32)
  KERNEL_DIPU2(nuclear_norm, dim, fp32)
  KERNEL_DIPU(cosine_similarity, fp32)
  KERNEL_DIPU(poisson_nll_loss, fp32)
  KERNEL_DIPU(cosine_embedding_loss, fp32)
  KERNEL_DIPU(nll_loss, fp32)
  KERNEL_DIPU(nll_loss2d, fp32)
  KERNEL_DIPU(hinge_embedding_loss, fp32)
  KERNEL_DIPU(kl_div, fp32)
  KERNEL_DIPU(l1_loss, fp32)
  KERNEL_DIPU(smooth_l1_loss, fp32)
  KERNEL_DIPU(huber_loss, fp32)
  KERNEL_DIPU(mse_loss, fp32)
  KERNEL_DIPU(margin_ranking_loss, fp32)
  KERNEL_DIPU(multilabel_margin_loss, fp32)
  KERNEL_DIPU(soft_margin_loss, fp32)
  KERNEL_DIPU(triplet_margin_loss, fp32)
  KERNEL_DIPU(multi_margin_loss, fp32)
  KERNEL_DIPU(binary_cross_entropy_with_logits, fp32)
  KERNEL_DIPU(dist, fp32)
  KERNEL_DIPU(pdist, fp32)
  KERNEL_DIPU(cdist, fp32)
  KERNEL_DIPU(renorm, fp32)
  KERNEL_DIPU(logsumexp, fp32)
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
      at::norm, "norm.Scalar", Tensor(const Tensor &, const Scalar &),
      Tensor(const Tensor &, const c10::optional<Scalar> &, ScalarType),
      fp32_append_dtype)
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(
      at::norm, "norm.ScalarOpt_dim",
      Tensor(const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool),
      Tensor(const Tensor &, const c10::optional<Scalar> &, IntArrayRef, bool,
             ScalarType),
      fp32_append_dtype)
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(
      at::norm, "norm.names_ScalarOpt_dim",
      Tensor(const Tensor &, const c10::optional<Scalar> &, DimnameList, bool),
      Tensor(const Tensor &, const c10::optional<Scalar> &, DimnameList, bool,
             ScalarType),
      fp32_append_dtype)
  // promote
  KERNEL_DIPU(addcdiv, promote)
  KERNEL_DIPU(addcmul, promote)
  KERNEL_DIPU(atan2, promote)
  KERNEL_DIPU(bilinear, promote)
  KERNEL_DIPU(cross, promote)
  KERNEL_DIPU(dot, promote)
  KERNEL_DIPU(grid_sampler, promote)
  KERNEL_DIPU(index_put, promote)
  KERNEL_DIPU(tensordot, promote)
  KERNEL_DIPU(scatter_add, promote)

  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
         TORCH_FN((&at::autocast::DipuBinaryCrossEntropyBanned)));
}

}  // namespace

}  // namespace autocast
}  // namespace at
