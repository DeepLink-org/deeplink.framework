

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

#define DIPU_BINARY_OP_CONFIG() InferConfig()

#define DIPU_BINARY_FLOAT_OP_CONFIG() \
  InferConfig().set_promote_integer_inputs_to_float(true)

#define INFER_NAME(name, overload) Infer_##name##_##overload

#define DIPU_INFER_STRUCT(name, overload) \
  struct INFER_NAME(name, overload) final : public Infer

#define DIPU_TORCH_META_FUNC(name, overload) \
  void INFER_NAME(name, overload)::meta

namespace dipu {
namespace native {

struct InferConfig {
  InferConfig& set_promote_integer_inputs_to_float(bool val) {
    promote_integer_inputs_to_float_ = val;
    return *this;
  }

  bool promote_integer_inputs_to_float_ = false;
};
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
  void compute_shape();         // compute the output shape

  c10::SmallVector<const at::Tensor*, 4> p_tensors_ = {};
  c10::DimVector target_shape_ = {};
  at::ScalarType common_dtype_ = at::ScalarType::Undefined;
  InferConfig config_{};
};

DIPU_INFER_STRUCT(add, Tensor) {
  void meta(const at::Tensor& self, const at::Tensor& other,
            const at::Scalar& alpha);
};

DIPU_INFER_STRUCT(sub, Tensor) {
  void meta(const at::Tensor& self, const at::Tensor& other,
            const at::Scalar& alpha);
};

DIPU_INFER_STRUCT(mul, Tensor) {
  void meta(const at::Tensor& self, const at::Tensor& other);
};

DIPU_INFER_STRUCT(div, Tensor) {
  void meta(const at::Tensor& self, const at::Tensor& other);
};

}  // namespace native
}  // namespace dipu
