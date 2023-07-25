// Copyright (c) 2023, DeepLink.
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/native/CPUFallback.h>
#include "DIPUATenFunctions.h"

#include <diopi/functions.h>

#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/Log.h>

using dnative = dipu::native::DIPUATenFunctions;

namespace dipu {

bool get_force_fallback(const char* opname);

};  // namespace dipu

namespace at {

 void dipu_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack);

#define DIPU_REGISTER_LOG(x)                                \
    {                                                       \
        const char* env = std::getenv("DIPU_DUMP_OP_ARGS"); \
        if (env != nullptr && std::atoi(env) > 0) {         \
            std::cout << x;                                 \
        }                                                   \
    }

// Temporarily not implement 'sub-dispatch from box' (from torch box func -> ourself unbox func)
// which described in design doc.
// because: 1. it need many add type trait code. 2. pytorch seems are sorting out infer and other pre/post code.
// so we shouldn't created a new preprocess logic?
//so just do a simple runtime cpu fallback to support diopi func loss
#define DIOPI_ATEN_FUNC(opname, diopiFunc, wapperFunc) do {                                                             \
    if ((reinterpret_cast<void*>(diopiFunc) != nullptr) && (!dipu::get_force_fallback(opname))) {                       \
        m.impl(opname, TORCH_FN(wapperFunc));                                                                           \
    }  else {                                                                                                           \
        if ((reinterpret_cast<void*>(diopiFunc) == nullptr)) {                                                          \
            DIPU_REGISTER_LOG(#diopiFunc << " is not yet implemented, ");                                               \
        } else {                                                                                                        \
            DIPU_REGISTER_LOG("force fallback has been set, ");                                                         \
        }                                                                                                               \
        DIPU_REGISTER_LOG(opname << " will be fallback to cpu" << std::endl);                                           \
        m.impl(opname, torch::CppFunction::makeFromBoxedFunction<&dipu_fallback>());                                    \
    }                                                                                                                   \
} while (false);

#define DIOPI_ATEN_FUNC_CUSTOM_FALLBACK(opname, diopi_func, force_fallback, wapper_func, custom_fallback_func) do {     \
    if ((reinterpret_cast<void*>(diopi_func) != nullptr) && !(force_fallback || dipu::get_force_fallback(opname))) {    \
        m.impl(opname, TORCH_FN(wapper_func));                                                                          \
    }  else {                                                                                                           \
        if ((reinterpret_cast<void*>(diopi_func) == nullptr)) {                                                         \
            DIPU_REGISTER_LOG(#diopi_func << " is not yet implemented, ")  ;                                            \
        } else {                                                                                                        \
            DIPU_REGISTER_LOG("force fallback has been set, ");                                                         \
        }                                                                                                               \
        DIPU_REGISTER_LOG(opname << " will be fallback to cpu" << std::endl);                                           \
        m.impl(opname, TORCH_FN(custom_fallback_func));                                                                 \
    }                                                                                                                   \
} while (false);

} //end ns at
