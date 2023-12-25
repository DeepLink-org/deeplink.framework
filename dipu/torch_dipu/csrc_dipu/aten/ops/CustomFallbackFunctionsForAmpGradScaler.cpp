// Copyright (c) 2023, DeepLink.
//
// This file contains definitions of custom fallback functions needed by AMP
// GradScaler. The corresponding declarations can be found in
// CustomFallbackFunctions.hpp.

#include <ATen/ATen.h>

#include "csrc_dipu/aten/RegisterDIPU.hpp"

namespace dipu {
namespace native {

namespace {

void _amp_non_finite_check_and_unscale_(at::Tensor& scaled_grad,
                                        at::Tensor& found_inf,
                                        const at::Tensor& inv_scale) {
  scaled_grad *= inv_scale.item();
  if (!scaled_grad.isfinite().all().item<bool>()) {
    found_inf[0] = 1.F;
  }
}

}  // anonymous namespace

// Multiplies each tensor in scaled_grads by inv_scale in-place.
// If any element of any tensor in scaled_grads is inf or NaN, sets found_inf
// to 1.0.
//
// Args:
// scaled_grads  A TensorList of scaled gradient tensors. May contain infs or
//               NaNs.
// found_inf     A single-element float tensor to which 1.0 will be written
//               if any gradient contain infs/nans. Pre-zeroing found_inf, if
//               appropriate, is the responsibility of the caller.
// inv_scale     The inverse of the scale factor by which scaled_grads are
//               currently multiplied.
void custom_fallback_dipu__amp_foreach_non_finite_check_and_unscale_(
    at::TensorList scaled_grads, at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  DIPU_OP_LOG_WARNING_ONCE(
      "custom fallback to separated ops, "
      "name=_amp_foreach_non_finite_check_and_unscale_"
      << std::endl);
  TORCH_CHECK(inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  for (const at::Tensor& t : scaled_grads) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): const_cast here is safe according to pytorch's source code
    _amp_non_finite_check_and_unscale_(const_cast<at::Tensor&>(t), found_inf,
                                       inv_scale);
  }
}

// Updates the scale tensor in place.
//
// Args:
// current_scale    A one-element cuda float tensor containing the scale value.
// growth_tracker   A one-element torch.cuda.IntTensor containing the number of
//                  recent consecutive unskipped steps.
// found_inf        A one-element cuda float tensor. If > 0, indicates that
//                  infs/nans were found by the relevant prior
//                  _amp_non_finite_check_and_unscale_cuda call, and 0 if no
//                  infs/nans were found.
// growth_factor    Multiplier if no infs/NaNs were found
//                  (typically slightly > 1).
// backoff_factor   Multiplier if infs/NaNs were found (typically 0.5).
// growth_interval  Number of consecutive unskipped steps that must occur for
//                  current_scale to be multiplied by growth_factor.
//
// Returns:
// current_scale
at::Tensor& custom_fallback_dipu__amp_update_scale_(at::Tensor& current_scale,
                                                    at::Tensor& growth_tracker,
                                                    const at::Tensor& found_inf,
                                                    double growth_factor,
                                                    double backoff_factor,
                                                    int64_t growth_interval) {
  DIPU_OP_LOG_WARNING_ONCE(
      "custom fallback to separated ops, name=_amp_update_scale_" << std::endl);
  TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int,
              "growth_tracker must be an int tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float,
              "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float,
              "found_inf must be a float tensor.");
  if (static_cast<bool>(found_inf.item<float>())) {
    current_scale *= backoff_factor;
    growth_tracker[0] = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = growth_tracker.item<int>() + 1;
    if (successful == growth_interval) {
      current_scale *= growth_factor;
      growth_tracker[0] = 0;
    } else {
      growth_tracker[0] = successful;
    }
  }
  return current_scale;
}

}  // namespace native
}  // namespace dipu
