#include <cstdint>
#include <iostream>

#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ops/add_meta.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/aten/ops/DIPUAmp.hpp"

#define DIPU_BINARY_OP_CONFIG() InferConfig()

#define DIPU_TORCH_META_FUNC2(name, overload) \
  void Infer_##name##_##overload::meta

namespace dipu {
namespace native {

class InferConfig {};
class Infer {
 public:
  Infer& add_input(const at::Tensor* p_tensor);
  Infer& set_config(const InferConfig& config);
  void build();

  // get
  c10::IntArrayRef target_shape() { return target_shape_; }
  at::ScalarType common_dtype() { return common_dtype_; }

 private:
  void compute_common_dtype();  // compute the output dtype
  void compute_shape();  // compute the output shape

  c10::SmallVector<const at::Tensor*, 4> p_tensors_ = {};
  c10::IntArrayRef target_shape_ = {};
  at::ScalarType common_dtype_ = at::ScalarType::Undefined;
  InferConfig config_{};
};
struct Infer_add_Tensor final : public Infer {
  void meta(const at::Tensor& self, const at::Tensor& other,
            const at::Scalar& alpha);
};

}  // namespace native
}  // namespace dipu
