#include <csrc_dipu/aten/ops/DIPUAmp.hpp>

namespace at {
namespace autocast {
namespace{

TORCH_LIBRARY_IMPL(_, AutocastDIPU, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastDIPU, m) {
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
  // commenting these out because they accept an explicit (not-optional) dtype, and we shouldn't try to flip that even
  // when autocasting.
  // KERNEL_DIPU2(norm, ScalarOpt_dtype, fp32_set_opt_dtype)
  // KERNEL_DIPU2(norm, ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  // KERNEL_DIPU2(norm, names_ScalarOpt_dim_dtype, fp32_set_opt_dtype)
  KERNEL_DIPU(sum, fp32_set_opt_dtype)
  KERNEL_DIPU2(sum, dim_IntList, fp32_set_opt_dtype)
  KERNEL_DIPU2(sum, dim_DimnameList, fp32_set_opt_dtype)
  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.
  // norm does not implicitly promote, but be aware when adding new ops to this policy.
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.Scalar", Tensor (const Tensor &, const Scalar&), Tensor (const Tensor &, const c10::optional<Scalar>&, ScalarType), fp32_append_dtype)
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, IntArrayRef, bool, ScalarType), fp32_append_dtype)
  KERNEL_DIPU_DIFFERENT_REDISPATCH_SIGNATURE(ADD_NS(norm), "norm.names_ScalarOpt_dim", Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool), Tensor (const Tensor &, const c10::optional<Scalar>&, DimnameList, bool, ScalarType), fp32_append_dtype)
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
      TORCH_FN((&at::autocast::dipu_binary_cross_entropy_banned)));
}

} // namespace

} // namespace autocast
} // namespace at