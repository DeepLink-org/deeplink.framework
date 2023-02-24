#include "./common.h"

namespace dipu {
bool isDeviceTensor(const at::Tensor &tensor) {
  return tensor.unsafeGetTensorImpl()->device_type() == dipu::DIPU_DEVICE_TYPE;
}
}