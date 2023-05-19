// Copyright (c) 2023, DeepLink.
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/native/CPUFallback.h>
#include "DIPUATenFunctions.h"

#include <diopi/functions.h>

#include <csrc_dipu/runtime/rthelper.h>
#include "util/Log.h"

using dnative = dipu::native::DIPUATenFunctions;

namespace at {

 void dipu_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack);


// Temporarily not implement 'sub-dispatch from box' (from torch box func -> ourself unbox func)
// which described in design doc.
// because: 1. it need many add type trait code. 2. pytorch seems are sorting out infer and other pre/post code.
// so we shouldn't created a new preprocess logic?
//so just do a simple runtime cpu fallback to support diopi func loss
#define DIOPI_ATEN_FUNC(opname, diopiFunc, wapperFunc) do {                                     \
    if (reinterpret_cast<void*>(diopiFunc) != nullptr) {                                        \
        m.impl(opname, TORCH_FN(wapperFunc));                                                   \
    }  else {                                                                                   \
        DIPU_LOG_ONCE << #diopiFunc << " is not yet implemented, "                              \
            << opname << " will be fallback to cpu" << std::endl;                               \
        m.impl(opname, torch::CppFunction::makeFromBoxedFunction<&dipu_fallback>());            \
    }                                                                                           \
} while (false);

} //end ns at
