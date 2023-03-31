#include "DIPUGuardImpl.h"

namespace dipu {

constexpr at::DeviceType DIPUGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(DIPU_DEVICE_TYPE_MACRO, DIPUGuardImpl);


}  // namespace torch_dipu
