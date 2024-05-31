// Copyright (c) 2023, DeepLink.
#include "csrc_dipu/aten/DIPUATenFunctions.h"

extern "C" thread_local int diopiNestedScopeDepth;

namespace dipu {
namespace native {
namespace dipu_aten {

at::Device dnativeTensorImpl::device_custom() const{
    if(diopiNestedScopeDepth)
        return {c10::DeviceType::CUDA, device_opt_->index()};
    return device_default();
}
// dipu native func
};  // namespace dipu_aten

}  // namespace native
}  // namespace dipu
