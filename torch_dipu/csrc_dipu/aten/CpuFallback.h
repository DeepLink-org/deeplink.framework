#pragma once

#include <ATen/native/CPUFallback.h>

namespace dipu {

void dipuCpuFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

}  // namespace dipu