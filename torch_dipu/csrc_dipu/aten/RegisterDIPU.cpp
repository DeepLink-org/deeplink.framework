// Copyright (c) 2023, DeepLink.
#include "RegisterDIPU.hpp"
#include <regex>
#include <iostream>
#include <c10/util/Exception.h>
#include <c10/core/Storage.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/EmptyTensor.h>

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/profiler/profiler.h>
#include <csrc_dipu/runtime/core/DIPUCopyInplace.h>

static std::string force_fallback_operators_list = []()-> std::string {
    std::ifstream stream(".dipu_force_fallback_op_list.config", std::ios_base::in | std::ios::binary);
    std::string content;
    const char* env = std::getenv("DIPU_FORCE_FALLBACK_OPS_LIST");
    if (env != nullptr) {
      content += env;
    }
    if (stream.is_open()) {
      while (!stream.eof()) {
        std::string line;
        stream >> line;
        content += "," + line;
      }
    }
    return content;
}();


namespace dipu {
bool get_force_fallback(const char* opname) {
  if (force_fallback_operators_list.size() <= 0 || opname == nullptr) {
    return false;
  } else {
    std::stringstream strstream(force_fallback_operators_list);
    std::string force_fallback_pattern;
    while(std::getline(strstream, force_fallback_pattern, ',')) {
      if (force_fallback_pattern.size() <= 0) {
        continue;
      }
      bool force_fallback = std::regex_match(opname, std::regex(force_fallback_pattern));
      if (force_fallback) {
        return true;
      }
    }
  }
  return false;
}

namespace native {
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);
}

}

namespace at {

void dipu_fallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  const auto name = c10::toString(op.operator_name());

  TORCH_CHECK(name.find("foreach") == std::string::npos,
    "Currently the foreach operator does not support fallback");

  DIPU_REGISTER_LOG("fallback to cpu, name=" << c10::toString(op.operator_name()) << std::endl);

  const static std::vector<std::string> custom_fallback_operators_list{
    "aten::native_batch_norm",
    "aten::native_batch_norm.out",
    "aten::native_batch_norm_backward",
  };
  auto iter = std::find(custom_fallback_operators_list.cbegin(), custom_fallback_operators_list.cend(), std::string(name));
  if (iter != custom_fallback_operators_list.cend()) {
    dipu::native::cpu_fallback(op, stack);
  } else {
    at::native::cpu_fallback(op, stack);
  }
}


namespace {
  // dipu native ops
  at::Tensor wrapper_DIPU_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt,
        c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt,
        c10::optional<at::MemoryFormat> memory_format_opt) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    const DeviceGuard device_guard(device_or_default(device_opt));
    return dnative::empty(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  }

  at::Tensor wrapper_CPU_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt,
        c10::optional<at::Device> device_opt,
        c10::optional<bool> pin_memory_opt,
        c10::optional<at::MemoryFormat> memory_format_opt) {
    return dnative::empty_cpu(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  }

  at::Tensor wrapper_DIPU_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    const DeviceGuard device_guard(device_or_default(device_opt));
    return dnative::empty_strided(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  }

  at::Tensor wrapper_CPU_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
    return dnative::empty_strided_cpu(size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  }

  at::Tensor& wrapper_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    static bool use_slow_copy = (std::getenv("DIPU_USE_SLOW_COPY") != nullptr);
    if (use_slow_copy) {
      return dnative::copy_(self, src, non_blocking);
    } else {
      return dipu::getDipuCopyInplace()->run(self, src, non_blocking);
    }
  }

  at::Tensor wrapper_DIPU___reshape_alias(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
  }

  // only used by cpu_fallback.
  at::Tensor wrapper_DIPU___copy_from_and_resize(const at::Tensor & self, const at::Tensor& dst) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    dst.resize_as_(self).copy_(self);
    return dst;
  }

  const at::Tensor& wrapper_resize_(const at::Tensor& self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    // add guard for device switch.
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    return dnative::resize_(self, size, memory_format);
  }

  at::Tensor wrapper_DIPU__as_strided(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset) {
      // No device check
    // DeviceGuard omitted
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    return at::native::as_strided_tensorimpl(self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride), storage_offset.has_value() ? c10::make_optional(storage_offset->expect_int()) : c10::nullopt);
  }

  at::Tensor wrapper_DIPU__view(const at::Tensor & self, c10::SymIntArrayRef size) {
    // No device check
    // DeviceGuard omitted
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
  }

  at::Tensor wrapper_DIPU__view_as_real(const at::Tensor & self) {
    // DeviceGuard omitted
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    return at::native::view_as_real(self);
  }

  at::Tensor wrapper_DIPU__view_as_complex(const at::Tensor & self) {
    // DeviceGuard omitted
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    return at::native::view_as_complex(self);
  }

  at::Tensor & wrapper_DIPU__zero_(at::Tensor & self) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::zero_(self);
  }

  // it's a view op, However it's not registered by RegisterCompositeExplicitAutograd.cpp,
  // but by cpu/cuda backend.
  at::Tensor wrapper_DIPU__unfold(const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
    // No device check
    // DeviceGuard omitted
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    return at::native::unfold(self, dimension, size, step);
  }

  at::Scalar wrapper_DIPU___local_scalar_dense(const at::Tensor & self) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    const OptionalDeviceGuard device_guard(device_of(self));
    return dnative::_local_scalar_dense_dipu(self);
  }

  at::Tensor& wrapper_DIPU_source_Storage_set_(at::Tensor& self, at::Storage source) {
    // No device check
    // DeviceGuard omitted
    int64_t new_size = static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
    return dnative::set_storage_dipu_(self, std::move(source), 0, new_size, {});
  }

  at::Tensor& wrapper_DIPU_source_Storage_offset_set_(at::Tensor& self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
    // No device check
    // DeviceGuard omitted
    return dnative::set_storage_dipu_(self, source, storage_offset.expect_int(), C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
  }

  at::Tensor & wrapper_DIPU_source_Tensor_set_(at::Tensor& self, const at::Tensor & source) {
    // No device check
    // DeviceGuard omitted
    if (self.unsafeGetTensorImpl() != source.unsafeGetTensorImpl()) {
      return dnative::set_storage_dipu_(self, source.storage(), source.storage_offset(), source.sizes(), source.strides());
    }
    return self;
  }

  at::Tensor& wrapper_DIPU__set_(at::Tensor & self) {
    c10::optional<Device> common_device = nullopt;
    (void)common_device; // Suppress unused variable warning
    c10::impl::check_and_update_common_device(common_device, self, "wrapper_DIPU__set_", "self");
    const OptionalDeviceGuard device_guard(device_of(self));
    return dnative::set_dipu_(self);
  }

  bool wrapper_DIPU__is_set_to(const at::Tensor& self, const at::Tensor& tensor) {
    // No device check
    // DeviceGuard omitted
    return at::native::is_set_to(self, tensor);
  }

  bool wrapper_BackendSelect_is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
      // Only CPU tensors can be pinned
    if (!self.is_cpu()) {
      return false;
    }

    c10::DispatchKeySet dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(dipu::DIPU_DEVICE_TYPE)));
    return at::_ops::is_pinned::redispatch(dk, self, device);
  }

  at::Tensor wrapper_BackendSelect__pin_memory(const at::Tensor& self, c10::optional<at::Device> device) {
    TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
    c10::DispatchKeySet dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(dipu::DIPU_DEVICE_TYPE)));
    return at::_ops::_pin_memory::redispatch(dk, self, device);
  }

  bool wrapper_DIPU_is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    const OptionalDeviceGuard device_guard(device_of(self));
    return dnative::is_pinned(self, device);
  }

  at::Tensor wrapper_DIPU__pin_memory(const at::Tensor& self, c10::optional<at::Device> device) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    const OptionalDeviceGuard device_guard(device_of(self));
    return dnative::_pin_memory(self, device);
  }

  void wrapper_DIPU__record_stream(at::Tensor & self, at::Stream s) {
    dipu::profile::RecordBlockCreator dipu_recorder(__FUNCTION__);
    const OptionalDeviceGuard device_guard(device_of(self));
    dipu::recordStream(self.storage().data_ptr(), dipu::DIPUStream(s));
  }

}  // inner anonymous namespace


TORCH_LIBRARY_IMPL(_, DIPU_DEVICE_TYPE_MACRO, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&dipu_fallback>());
}

// Change to use XPU which already register this fallback in ATen/core/VariableFallbackKernel.cpp
// TORCH_LIBRARY_IMPL(_, DIPU_AUTOGRAD_DEVICE_TYPE_MACRO, m) {
//   m.fallback(torch::CppFunction::makeFallthrough());
// }

TORCH_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
  // always registered
  m.impl("empty.memory_format", TORCH_FN(wrapper_DIPU_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(wrapper_DIPU_empty_strided));
  m.impl("copy_",  TORCH_FN(wrapper_copy_));
  m.impl("_reshape_alias", TORCH_FN(wrapper_DIPU___reshape_alias));
  m.impl("_copy_from_and_resize", TORCH_FN(wrapper_DIPU___copy_from_and_resize));
  m.impl("resize_", TORCH_FN(wrapper_resize_));
  m.impl("as_strided", TORCH_FN(wrapper_DIPU__as_strided));
  m.impl("view", TORCH_FN(wrapper_DIPU__view));
  m.impl("view_as_real", TORCH_FN(wrapper_DIPU__view_as_real));
  m.impl("view_as_complex", TORCH_FN(wrapper_DIPU__view_as_complex));
  m.impl("zero_", TORCH_FN(wrapper_DIPU__zero_));
  m.impl("unfold", TORCH_FN(wrapper_DIPU__unfold));
  m.impl("_local_scalar_dense", TORCH_FN(wrapper_DIPU___local_scalar_dense));
  m.impl("set_.source_Storage", TORCH_FN(wrapper_DIPU_source_Storage_set_));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(wrapper_DIPU_source_Storage_offset_set_));
  m.impl("set_.source_Tensor", TORCH_FN(wrapper_DIPU_source_Tensor_set_));
  m.impl("set_", TORCH_FN(wrapper_DIPU__set_));
  m.impl("is_set_to", TORCH_FN(wrapper_DIPU__is_set_to));
  m.impl("is_pinned", TORCH_FN(wrapper_DIPU_is_pinned));
  m.impl("_pin_memory", TORCH_FN(wrapper_DIPU__pin_memory));
  m.impl("record_stream", TORCH_FN(wrapper_DIPU__record_stream));
}

class IgnoreWarningHandler : public c10::WarningHandler {
public:
  void process(const c10::Warning& warning) {
    // do nothing
  }
};

c10::WarningHandler* getIgnoreHandler() {
  static IgnoreWarningHandler handler_ = IgnoreWarningHandler();
  return &handler_;
}

// override BackendSelect is_pinned and _pin_memory operator
TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  // disable override warning log which like
  // [W OperatorEntry.cpp:159] Warning: Overriding a previously registered kernel for the same operator and the same dispatch key
  c10::WarningUtils::WarningHandlerGuard guard(getIgnoreHandler());
  m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(wrapper_BackendSelect_is_pinned));
  m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(wrapper_BackendSelect__pin_memory));
}

// override CPU operator
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  // disable override warning log
  c10::WarningUtils::WarningHandlerGuard guard(getIgnoreHandler());
  m.impl("empty.memory_format", TORCH_FN(wrapper_CPU_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(wrapper_CPU_empty_strided));
}

}  //end ns at
