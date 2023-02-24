#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/native/CPUFallback.h>
#include "DIPUNativeFunctions.h"
#include <csrc_dipu/runtime/rthelper.h>

namespace dnative = dipu::native;
namespace at { 
namespace {

  at::Tensor wrapper_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt,
        c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt,
        c10::optional<at::MemoryFormat> memory_format_opt) {
    return dnative::empty(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  }

  at::Tensor wrapper_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
    return dnative::empty_strided(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  } 

  at::Tensor& wrapper_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    return dnative::copy_(self, src, non_blocking);
  }

  at::Tensor wrapper_DIPU___reshape_alias(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
    return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
  }

  // only used by cpu_fallback.
  at::Tensor wrapper_DIPU___copy_from_and_resize(const at::Tensor & self, const at::Tensor& dst) {
    at::Tensor ret = dnative::copy_(const_cast<at::Tensor& >(dst), self, false);
    return ret;
  }

}  // anonymous ns

static void dipu_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  const auto name = c10::toString(op.operator_name());
  DIPU_LOGE("fallback %s \n", name);
  at::native::cpu_fallback(op, stack);
}

void dummyEmptyfunc(){
}

// Temporarily not implement 'sub-dispatch from box' (from torch box func -> ourself unbox func)
// which described in design doc.
// because: 1. it need many add type trait code. 2. pytorch seems are sorting out infer and other pre/post code.
// so we shouldn't created a new preprocess logic?
//so just do a simple runtime cpu fallback to support diopi func loss
#define ADD_DIPU_ATEN_FUNC(ATEN_NAME, FUNC_NAME) \
{ \
  bool isdiopiOp = false; \
  if (isdiopiOp) {   \
    m.impl(ATEN_NAME, TORCH_FN(FUNC_NAME)); \
  }  \
  else {  \
    m.impl(ATEN_NAME, torch::CppFunction::makeFromBoxedFunction<&dipu_fallback>());  \
  } \
} \

TORCH_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
  m.impl("empty.memory_format", TORCH_FN(wrapper_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(wrapper_empty_strided));
  m.impl("copy_",  TORCH_FN(wrapper_copy_));
  m.impl("_reshape_alias", TORCH_FN(wrapper_DIPU___reshape_alias));
  m.impl("_copy_from_and_resize", TORCH_FN(wrapper_DIPU___copy_from_and_resize));

  ADD_DIPU_ATEN_FUNC("add.out", dummyEmptyfunc);
}


} //end ns at