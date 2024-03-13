#include <ATen/native/BinaryOps.h>
#include <ATen/TensorMeta.h>
#include <ATen/ops/add_meta.h>
#include <ATen/NamedTensorUtils.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"

#define DIPU_BINARY_OP_CONFIG()              \
  at::TensorIteratorConfig()                     \
      .set_check_mem_overlap(true)           \
      .allow_cpu_scalars(true)               \
      .promote_inputs_to_common_dtype(false) \
      .cast_common_dtype_to_outputs(false)   \
      .enforce_safe_casting_to_output(false)

#define DIPU_TORCH_META_FUNC2(name, overload) \
  void Infer_##name##_##overload::meta

at::Tensor create_out(at::IntArrayRef sizes, at::IntArrayRef strides,
                      const c10::TensorOptions& options) {
  if (strides.empty()) {
    return dipu::native::dipu_aten::empty(
        sizes, options.dtype_opt()->toScalarType(), options.layout_opt(), options.device_opt(),
        options.pinned_memory_opt(), options.memory_format_opt());
  } else {
    return dipu::native::dipu_aten::empty_strided(
        sizes, strides, options.dtype_opt()->toScalarType(), options.layout_opt(),
        options.device_opt(), options.pinned_memory_opt());
  }
}

struct Infer_add_Tensor final : public at::meta::structured_add_Tensor {
  void meta(const at::Tensor& self, const at::Tensor& other,
            const at::Scalar& alpha);

  void set_output_strided(
        int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
        c10::TensorOptions options, at::DimnameList names
    ) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          at::namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::meta::structured_add_Tensor::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
    void set_output_raw_strided(
        int64_t output_idx, at::IntArrayRef sizes, at::IntArrayRef strides,
        c10::TensorOptions options, at::DimnameList names
    ) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        if (!names.empty()) {
          at::namedinference::propagate_names(outputs_[output_idx], names);
        }
         // super must happen after, so that downstream can use maybe_get_output
         // to retrieve the output
        at::meta::structured_add_Tensor::set_output_raw_strided(output_idx, sizes, strides, options, names);
    }
  const at::Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx];
  }
  const at::Tensor& maybe_get_output() {
    return maybe_get_output(0);
  }
  std::array<at::Tensor, 1> outputs_;
};

DIPU_TORCH_META_FUNC2(add, Tensor)
(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  build(DIPU_BINARY_OP_CONFIG()
            .add_output(maybe_get_output())
            .add_input(self)
            .add_input(other));
  at::native::alpha_check(dtype(), alpha);
}
