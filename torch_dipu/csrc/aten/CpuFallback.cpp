#include "torch_dipu/csrc/aten/CpuFallback.h"
#include "torch_dipu/csrc/aten/util/Log.h"

#include <iostream>

namespace dipu {

void dipuCpuFallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    DIPU_LOG_ONCE << "fallback to cpu, name=" << c10::toString(op.operator_name()) << std::endl;
    // Call the actual boxed CPU fallback.
    at::native::cpu_fallback(op, stack);
}

// TORCH_LIBRARY_IMPL(_, XLA, m) {
//     m.fallback(torch::CppFunction::makeFromBoxedFunction<&dipuCpuFallback>());
// }

}  // namespace dipu