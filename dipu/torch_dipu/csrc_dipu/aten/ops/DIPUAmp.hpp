// Copyright (c) 2023, DeepLink.
//
// This file contains user-customizable autocast policies for Automatic Mixed
// Precision (AMP).
//
// Each vendor should provide a "vendor_autocast.h" header containing
// vendor-specified autocast policies that would override the default ones.
//
// Example:
//   #pragma once
//
//   #include "csrc_dipu/aten/ops/DIPUAmp.hpp"
//
//   namespace dipu {
//   namespace autocast {
//
//   DIPU_CUSTOMIZE_OP_CAST_POLICY(dot, kLowerPrecisionFp);
//   DIPU_CUSTOMIZE_OP_CAST_POLICY(conv1d, kFp32);
//   DIPU_CUSTOMIZE_OP_CAST_POLICY(conv2d, kFp32);
//   DIPU_CUSTOMIZE_OP_CAST_POLICY(conv3d, kFp32);
//   DIPU_CUSTOMIZE_OP_CAST_POLICY(mm, kPromote);
//
//   }  // namespace autocast
//   }  // namespace dipu
//
// In this example,
// - dot will run in lower precision (float16, bfloat16, etc.);
// - conv1d, conv2d, conv3d will run in float32;
// - mm will run in the widest dtype among args;
// - the other ops will run in default policies.
//
// If no "vendor_autocast.h" or an empty one is provided, all ops will run in
// default policies (just like CUDA).
//
// Go through this file for available cast policies, the list of customizable
// ops and their default policies.
//
// See also: https://pytorch.org/docs/stable/amp.html

#pragma once

#include <cstdint>

namespace dipu {
namespace autocast {

// Cast policies of ops' input and output within autocast block.
enum class CastPolicy : std::uint8_t {
  kInvalid = 0,
  kLowerPrecisionFp,  // cast into user-configurable lower precision
                      // floating-point type (e.g. float16, bfloat16).
  kFp32,              // cast into float32.
  kPromote,           // run in the widest dtype among several args.
};

namespace details {

template <class Op, typename = void>
struct OpCastPolicyHelper {};

}  // namespace details

// Define OP as customizable, and set default policy.
// MUST be used in namespace dipu::autocast
#define DIPU_DEFAULT_OP_CAST_POLICY(OP, POLICY)           \
  namespace ops {                                         \
  struct OP {};                                           \
  }                                                       \
  template <typename _dummy>                              \
  struct details::OpCastPolicyHelper<ops::OP, _dummy> {   \
    static const CastPolicy kPolicy = CastPolicy::POLICY; \
  };

// Set custom cast policy for OP.
// MUST be used in namespace dipu::autocast
#define DIPU_CUSTOMIZE_OP_CAST_POLICY(OP, POLICY)         \
  template <>                                             \
  struct details::OpCastPolicyHelper<ops::OP, void> {     \
    static const CastPolicy kPolicy = CastPolicy::POLICY; \
  };

// Query for the final cast policy of OP.
// can be used anywhere
#define DIPU_OP_CAST_POLICY(OP)                                            \
  ::dipu::autocast::details::OpCastPolicyHelper<::dipu::autocast::ops::OP, \
                                                void>::kPolicy

// ---------------------------------------------------------------------------
// Default op policy settings begin from here.
// ONLY listed ops are customizable via DIPU_CUSTOMIZE_OP_CAST_POLICY.
// The other ops all run in fp32.
// ---------------------------------------------------------------------------

// lower_precision_fp
DIPU_DEFAULT_OP_CAST_POLICY(_convolution, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(conv1d, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(conv2d, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(conv3d, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(conv_tbc, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(conv_transpose1d, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(conv_transpose2d, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(conv_transpose3d, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(convolution, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(cudnn_convolution, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(cudnn_convolution_transpose, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(prelu, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(addmm, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(addmv, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(addr, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(matmul, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(einsum, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(mm, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(mv, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(linear, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(addbmm, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(baddbmm, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(bmm, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(chain_matmul, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(linalg_multi_dot, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(_thnn_fused_lstm_cell, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(_thnn_fused_gru_cell, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(lstm_cell, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(gru_cell, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(rnn_tanh_cell, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(rnn_relu_cell, kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(_scaled_dot_product_flash_attention,
                            kLowerPrecisionFp);
DIPU_DEFAULT_OP_CAST_POLICY(scaled_dot_product_attention, kLowerPrecisionFp);

// fp32
DIPU_DEFAULT_OP_CAST_POLICY(acos, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(asin, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(cosh, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(erfinv, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(exp, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(expm1, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(log, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(log10, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(log2, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(log1p, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(reciprocal, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(rsqrt, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(sinh, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(tan, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(pow, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(softplus, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(layer_norm, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(native_layer_norm, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(group_norm, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(frobenius_norm, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(nuclear_norm, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(cosine_similarity, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(poisson_nll_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(cosine_embedding_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(nll_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(nll_loss2d, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(hinge_embedding_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(kl_div, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(l1_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(smooth_l1_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(huber_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(mse_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(margin_ranking_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(multilabel_margin_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(soft_margin_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(triplet_margin_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(multi_margin_loss, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(binary_cross_entropy_with_logits, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(dist, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(pdist, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(cdist, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(renorm, kFp32);
DIPU_DEFAULT_OP_CAST_POLICY(logsumexp, kFp32);

// promote
DIPU_DEFAULT_OP_CAST_POLICY(addcdiv, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(addcmul, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(atan2, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(bilinear, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(cross, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(dot, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(grid_sampler, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(index_put, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(tensordot, kPromote);
DIPU_DEFAULT_OP_CAST_POLICY(scatter_add, kPromote);

// ------------------------ DEFAULT POLICIES ENDED ---------------------------

#undef DIPU_DEFAULT_OP_CAST_POLICY

}  // namespace autocast
}  // namespace dipu
