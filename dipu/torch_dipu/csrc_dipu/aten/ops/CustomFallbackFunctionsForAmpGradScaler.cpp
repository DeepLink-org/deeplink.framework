#include <ATen/ATen.h>

namespace dipu {
namespace native {

at::Tensor& dipu_mul__scalar(at::Tensor& self, const at::Scalar& other);

namespace {

void _amp_non_finite_check_and_unscale_(at::Tensor& scaled_grad,
                                        at::Tensor& found_inf,
                                        const at::Tensor& inv_scale) {
    scaled_grad *= inv_scale.item();
    if (scaled_grad.isinf().any().item<bool>()) {
        found_inf[0] = 1.f;
    }
}

}  // anonymous namespace

void custom_fallback_dipu__amp_foreach_non_finite_check_and_unscale_(
    at::TensorList scaled_grads, at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
    TORCH_CHECK(inv_scale.numel() == 1,
                "inv_scale must be a 1-element tensor.");
    TORCH_CHECK(found_inf.numel() == 1,
                "found_inf must be a 1-element tensor.");
    for (const at::Tensor& t : scaled_grads) {
        // const_cast here is safe according to pytorch's source code
        _amp_non_finite_check_and_unscale_(const_cast<at::Tensor&>(t),
                                           found_inf, inv_scale);
    }
}

at::Tensor& custom_fallback_dipu__amp_update_scale_(at::Tensor& current_scale,
                                                    at::Tensor& growth_tracker,
                                                    const at::Tensor& found_inf,
                                                    double growth_factor,
                                                    double backoff_factor,
                                                    int64_t growth_interval) {
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
