// Copyright (c) 2023, DeepLink.
#include "RegisterDIPU.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ATen/DeviceGuard.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/core/operator_name.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_pin_memory_ops.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/is_pinned_ops.h>
#include <ATen/ops/is_set_to_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/zero_native.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/profiler/profiler.h"
#include "csrc_dipu/runtime/core/DIPUStream.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocatorUtils.h"
#include "csrc_dipu/utils/helpfunc.hpp"

namespace dnative = dipu::native::dipu_aten;

namespace dipu {

namespace native {
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);
}  // end of namespace native

void dump_fallback_op_args(const c10::OperatorHandle& op,
                           const torch::jit::Stack* stack) {
  static int level = []() {
    const char* env_ptr = std::getenv("DIPU_DUMP_OP_ARGS");
    return env_ptr ? std::atoi(env_ptr) : 0;
  }();

  if (level < 1) {
    return;
  }
  const auto name = c10::toString(op.operator_name());
  printf("--%-50s %-30s \n", ("[" + name + "]:").data(), "dipu_fallback");

  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);

  auto dumpTensor = [&](const at::Tensor& tensor) {
    if (tensor.defined()) {
      std::cout << "numel: " << tensor.numel() << ", sizes: " << tensor.sizes()
                << ", stride: " << tensor.strides()
                << ", is_view: " << tensor.is_view()
                << ", dtype: " << tensor.dtype()
                << ", device:" << tensor.device()
                << ", layout:" << tensor.layout() << ", requires_grad: "
                << (tensor.requires_grad() ? "true" : "false")
                << ", pinned_memory: "
                << (tensor.is_pinned() ? "true" : "false")
                << ", memory_format: " << tensor.suggest_memory_format()
                << ", data_ptr: " << tensor.data_ptr();
      if (level > 2) {
        std::cout << std::endl << tensor;
      }
    } else {
      std::cout << "undefined";
    }
  };

  for (const auto idx : c10::irange(arguments.size())) {
    std::cout << "\t" << name << ": \t" << schema_args[idx].name() << ": ";
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      const auto& tensor = ivalue.toTensor();
      dumpTensor(tensor);
      std::cout << std::endl;
    } else if (ivalue.isTensorList()) {
      const auto& tensorlist = ivalue.toTensorList();
      std::cout << std::endl;
      for (const auto& tensor : tensorlist) {
        std::cout << "\t";
        dumpTensor(tensor);
        std::cout << std::endl;
      }
    } else {
      std::cout << ivalue << std::endl;
    }
  }
}

}  // end of namespace dipu

namespace at {

void dipu_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys,
                   torch::jit::Stack* stack) {
  dipu::dump_fallback_op_args(op, stack);
  const auto name = c10::toString(op.operator_name());
  DIPU_OP_LOG_WARNING_ONCE("fallback to cpu, name=" << name << std::endl);

#if DIPU_TORCH_VERSION < 20100
  // TORCH_CHECK(name.find("foreach") == std::string::npos,
  //   "Currently the foreach operator does not support fallback: ", name);
  const bool forech_op = name.find("foreach") != std::string::npos;

  const static std::vector<std::string> custom_fallback_operators_list{
      "aten::native_batch_norm",
      "aten::native_batch_norm.out",
      "aten::native_batch_norm_backward",
  };
  auto iter =
      std::find(custom_fallback_operators_list.cbegin(),
                custom_fallback_operators_list.cend(), std::string(name));
  if (iter != custom_fallback_operators_list.cend() || forech_op) {
    dipu::native::cpu_fallback(op, stack);
  } else {
    at::native::cpu_fallback(op, stack);
  }
#else
  at::native::cpu_fallback(op, stack);
#endif
}

namespace {
// dipu native ops
at::Tensor wrapper_DIPU_empty_memory_format(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  return dnative::empty(size, dtype_opt, layout_opt, device_opt, pin_memory_opt,
                        memory_format_opt);
}

at::Tensor wrapper_CPU_empty_memory_format(
    at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
    c10::optional<bool> pin_memory_opt,
    c10::optional<at::MemoryFormat> memory_format_opt) {
  return dnative::empty_cpu(size, dtype_opt, layout_opt, device_opt,
                            pin_memory_opt, memory_format_opt);
}

at::Tensor wrapper_DIPU_empty_strided(at::IntArrayRef size,
                                      at::IntArrayRef stride,
                                      c10::optional<at::ScalarType> dtype_opt,
                                      c10::optional<at::Layout> layout_opt,
                                      c10::optional<at::Device> device_opt,
                                      c10::optional<bool> pin_memory_opt) {
  return dnative::empty_strided(size, stride, dtype_opt, layout_opt, device_opt,
                                pin_memory_opt);
}

at::Tensor wrapper_CPU_empty_strided(at::IntArrayRef size,
                                     at::IntArrayRef stride,
                                     c10::optional<at::ScalarType> dtype_opt,
                                     c10::optional<at::Layout> layout_opt,
                                     c10::optional<at::Device> device_opt,
                                     c10::optional<bool> pin_memory_opt) {
  return dnative::empty_strided_cpu(size, stride, dtype_opt, layout_opt,
                                    device_opt, pin_memory_opt);
}

at::Tensor wrapper_DIPU___reshape_alias(const at::Tensor& self,
                                        c10::SymIntArrayRef size,
                                        c10::SymIntArrayRef stride) {
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW(size),
                                    C10_AS_INTARRAYREF_SLOW(stride));
}

// only used by cpu_fallback.
at::Tensor wrapper_DIPU___copy_from_and_resize(const at::Tensor& self,
                                               const at::Tensor& dst) {
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  dst.resize_as_(self).copy_(self);
  return dst;
}

const at::Tensor& wrapper_resize_(
    const at::Tensor& self, at::IntArrayRef size,
    c10::optional<at::MemoryFormat> memory_format) {
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return dnative::resize_(self, size, memory_format);
}

at::Tensor wrapper_DIPU__as_strided(const at::Tensor& self,
                                    c10::SymIntArrayRef size,
                                    c10::SymIntArrayRef stride,
                                    c10::optional<c10::SymInt> storage_offset) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return at::native::as_strided_tensorimpl(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride),
      storage_offset.has_value()
          ? c10::make_optional(storage_offset->expect_int())
          : c10::nullopt);
}

at::Tensor wrapper_DIPU__view(const at::Tensor& self,
                              c10::SymIntArrayRef size) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor wrapper_DIPU__view_as_real(const at::Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return at::native::view_as_real(self);
}

at::Tensor wrapper_DIPU__view_as_complex(const at::Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return at::native::view_as_complex(self);
}

at::Tensor& wrapper_DIPU__zero_(at::Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return at::native::zero_(self);
}

// it's a view op, However it's not registered by
// RegisterCompositeExplicitAutograd.cpp, but by cpu/cuda backend.
at::Tensor wrapper_DIPU__unfold(const at::Tensor& self, int64_t dimension,
                                int64_t size, int64_t step) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return at::native::unfold(self, dimension, size, step);
}

at::Scalar wrapper_DIPU___local_scalar_dense(const at::Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return dnative::_local_scalar_dense_dipu(self);
}

// NOLINTBEGIN(performance-unnecessary-value-param)
at::Tensor& wrapper_DIPU_source_Storage_set_(at::Tensor& self,
                                             at::Storage source) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  int64_t new_size =
      static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  return dnative::set_storage_dipu_(self, std::move(source), 0, new_size, {});
}

at::Tensor& wrapper_DIPU_source_Storage_offset_set_(
    at::Tensor& self, at::Storage source, c10::SymInt storage_offset,
    c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  // No device check
  // DeviceGuard omitted
  return dnative::set_storage_dipu_(
      self, std::move(source), storage_offset.expect_int(),
      C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}
// NOLINTEND(performance-unnecessary-value-param)

at::Tensor& wrapper_DIPU_source_Tensor_set_(at::Tensor& self,
                                            const at::Tensor& source) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  if (self.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
    return dnative::set_storage_dipu_(self, source.storage(),
                                      source.storage_offset(), source.sizes(),
                                      source.strides());
  }
  return self;
}

at::Tensor& wrapper_DIPU__set_(at::Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device;  // Suppress unused variable warning
  c10::impl::check_and_update_common_device(common_device, self,
                                            "wrapper_DIPU__set_", "self");
  const OptionalDeviceGuard device_guard(device_of(self));
  return dnative::set_dipu_(self);
}

bool wrapper_DIPU__is_set_to(const at::Tensor& self, const at::Tensor& tensor) {
  // No device check
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::is_set_to(self, tensor);
}

bool wrapper_BackendSelect_is_pinned(const at::Tensor& self,
                                     c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }

  c10::DispatchKeySet dk = c10::DispatchKeySet(c10::computeDispatchKey(
      c10::nullopt, self.layout(), device.value_or(dipu::DIPU_DEVICE_TYPE)));
  return at::_ops::is_pinned::redispatch(dk, self, device);
}

at::Tensor wrapper_BackendSelect__pin_memory(const at::Tensor& self,
                                             c10::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(),
              "' only dense CPU tensors can be pinned");
  c10::DispatchKeySet dk = c10::DispatchKeySet(c10::computeDispatchKey(
      c10::nullopt, self.layout(), device.value_or(dipu::DIPU_DEVICE_TYPE)));
  return at::_ops::_pin_memory::redispatch(dk, self, device);
}

bool wrapper_DIPU_is_pinned(const at::Tensor& self,
                            c10::optional<at::Device> device) {
  dipu::DIPUGuard grard(dipu::DIPU_DEVICE_TYPE);
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return dnative::is_pinned(self, device);
}

at::Tensor wrapper_DIPU__pin_memory(const at::Tensor& self,
                                    c10::optional<at::Device> device) {
  dipu::DIPUGuard grard(dipu::DIPU_DEVICE_TYPE);
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  return dnative::_pin_memory(self, device);
}

void wrapper_DIPU__record_stream(at::Tensor& self, at::Stream s) {
  const OptionalDeviceGuard device_guard(device_of(self));
  dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
  dipu::recordStream(self.storage().data_ptr(), dipu::DIPUStream(s));
}

}  // namespace

DIPU_LIBRARY_IMPL(_, DIPU_DEVICE_TYPE_MACRO, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&dipu_fallback>());
}

// Change to use XPU which already register this fallback in
// ATen/core/VariableFallbackKernel.cpp TORCH_LIBRARY_IMPL(_,
// DIPU_AUTOGRAD_DEVICE_TYPE_MACRO, m) {
//   m.fallback(torch::CppFunction::makeFallthrough());
// }

DIPU_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
  // always registered
  m.impl("empty.memory_format", TORCH_FN(wrapper_DIPU_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(wrapper_DIPU_empty_strided));
  m.impl("_reshape_alias", TORCH_FN(wrapper_DIPU___reshape_alias));
  m.impl("_copy_from_and_resize",
         TORCH_FN(wrapper_DIPU___copy_from_and_resize));
  m.impl("resize_", TORCH_FN(wrapper_resize_));
  m.impl("as_strided", TORCH_FN(wrapper_DIPU__as_strided));
  m.impl("view", TORCH_FN(wrapper_DIPU__view));
  m.impl("view_as_real", TORCH_FN(wrapper_DIPU__view_as_real));
  m.impl("view_as_complex", TORCH_FN(wrapper_DIPU__view_as_complex));
  m.impl("zero_", TORCH_FN(wrapper_DIPU__zero_));
  m.impl("unfold", TORCH_FN(wrapper_DIPU__unfold));
  m.impl("_local_scalar_dense", TORCH_FN(wrapper_DIPU___local_scalar_dense));
  m.impl("set_.source_Storage", TORCH_FN(wrapper_DIPU_source_Storage_set_));
  m.impl("set_.source_Storage_storage_offset",
         TORCH_FN(wrapper_DIPU_source_Storage_offset_set_));
  m.impl("set_.source_Tensor", TORCH_FN(wrapper_DIPU_source_Tensor_set_));
  m.impl("set_", TORCH_FN(wrapper_DIPU__set_));
  m.impl("is_set_to", TORCH_FN(wrapper_DIPU__is_set_to));
  m.impl("is_pinned", TORCH_FN(wrapper_DIPU_is_pinned));
  m.impl("_pin_memory", TORCH_FN(wrapper_DIPU__pin_memory));
  m.impl("record_stream", TORCH_FN(wrapper_DIPU__record_stream));
}

/**
when test EasyLLM using dipu + muxi pytorch, some mem related error happened
(Invalid Address). so temporarily use this macro to disable dipu's related op
and to use muxi native cpu empty op to bypass this issue. see the doc below
written by guochun1 for detail.
https://aicarrier.feishu.cn/wiki/PXdYwcsjii9TJZk5AR1cG2bxnFd

Warnning: Need check if this bypass works/needed when using dipu + torch_cpu which not
contains 'mx' pin mem/cpu-empty op.
TODO: fandaoyi
**/
#if !DIPU_VENDOR_NAME_MUXI

DIPU_LIBRARY_IMPL(aten, BackendSelect, m) {
  c10::WarningUtils::WarningHandlerGuard guard(dipu::getIgnoreHandler());
  m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"),
         TORCH_FN(wrapper_BackendSelect_is_pinned));
  m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"),
         TORCH_FN(wrapper_BackendSelect__pin_memory));
}

// override CPU operator
DIPU_LIBRARY_IMPL(aten, CPU, m) {
  // disable override warning log
  c10::WarningUtils::WarningHandlerGuard guard(dipu::getIgnoreHandler());
  m.impl("empty.memory_format", TORCH_FN(wrapper_CPU_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(wrapper_CPU_empty_strided));
}

#endif

}  // end namespace at
