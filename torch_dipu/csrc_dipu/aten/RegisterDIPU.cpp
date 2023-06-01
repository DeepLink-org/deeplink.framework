// Copyright (c) 2023, DeepLink.
#include "RegisterDIPU.hpp"
#include <regex>

namespace dipu {

bool get_force_fallback(const char* opname) {
  static const char* force_fallback_operators_list_str = std::getenv("DIPU_FORCE_FALLBACK_OPS_LIST");
  if (force_fallback_operators_list_str == nullptr || opname == nullptr) {
    return false;
  } else {
    DIPU_LOG_ONCE << "env DIPU_FORCE_FALLBACK_OPS_LIST:" << force_fallback_operators_list_str << std::endl;

    const std::string pattern("(([;, ]+)|())(aten::)*" + std::string(opname) + "(([ ,;]+)|())");

    const bool matched_result = std::regex_search(force_fallback_operators_list_str, std::regex(pattern));
    if (matched_result) {
      DIPU_REGISTER_LOG << "since " << opname << " is in " << force_fallback_operators_list_str
        << ", forceed to fallback!" << std::endl;
      return true;
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

  std::cout << "fallback to cpu, name=" << c10::toString(op.operator_name()) << std::endl;

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
  at::Tensor wrapper_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
        c10::optional<at::Layout> layout_opt,
        c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt,
        c10::optional<at::MemoryFormat> memory_format_opt) {
    const DeviceGuard device_guard(device_or_default(device_opt));
    return dnative::empty(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, memory_format_opt);
  }

  at::Tensor wrapper_empty_strided(at::IntArrayRef size, at::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt,
      c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
    const DeviceGuard device_guard(device_or_default(device_opt));
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
    dst.resize_as_(self).copy_(self);
    return dst;
  }

  const at::Tensor& wrapper_resize_(const at::Tensor& self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    // add guard for device switch.
    return dnative::resize_(self, size, memory_format);
  }

  at::Tensor wrapper_DIPU__as_strided(const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<c10::SymInt> storage_offset) {
      // No device check
    // DeviceGuard omitted
    return at::native::as_strided_tensorimpl(self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride), storage_offset.has_value() ? c10::make_optional(storage_offset->expect_int()) : c10::nullopt);
  }

  at::Tensor wrapper_DIPU__view(const at::Tensor & self, c10::SymIntArrayRef size) {
    // No device check
    // DeviceGuard omitted
    return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
  }

  at::Tensor wrapper_DIPU__view_as_real(const at::Tensor & self) {
    // DeviceGuard omitted
    return at::native::view_as_real(self);
  }

  at::Tensor wrapper_DIPU__view_as_complex(const at::Tensor & self) {
    // DeviceGuard omitted
    return at::native::view_as_complex(self);
  }

  at::Tensor & wrapper_DIPU__zero_(at::Tensor & self) {
    const OptionalDeviceGuard device_guard(device_of(self));
    return at::native::zero_(self);
  }

  at::Scalar wrapper_DIPU___local_scalar_dense(const at::Tensor & self) {
    const OptionalDeviceGuard device_guard(device_of(self));
    return dnative::_local_scalar_dense_dipu(self);
  }

}  // inner anonymous namespace


TORCH_LIBRARY_IMPL(_, DIPU_DEVICE_TYPE_MACRO, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&dipu_fallback>());
}

TORCH_LIBRARY_IMPL(aten, DIPU_DEVICE_TYPE_MACRO, m) {
  // always registered
  m.impl("empty.memory_format", TORCH_FN(wrapper_empty_memory_format));
  m.impl("empty_strided", TORCH_FN(wrapper_empty_strided));
  m.impl("copy_",  TORCH_FN(wrapper_copy_));
  m.impl("_reshape_alias", TORCH_FN(wrapper_DIPU___reshape_alias));
  m.impl("_copy_from_and_resize", TORCH_FN(wrapper_DIPU___copy_from_and_resize));
  m.impl("resize_", TORCH_FN(wrapper_resize_));
  m.impl("as_strided", TORCH_FN(wrapper_DIPU__as_strided));
  m.impl("view", TORCH_FN(wrapper_DIPU__view));
  m.impl("view_as_real", TORCH_FN(wrapper_DIPU__view_as_real));
  m.impl("view_as_complex", TORCH_FN(wrapper_DIPU__view_as_complex));
  m.impl("zero_", TORCH_FN(wrapper_DIPU__zero_));
  m.impl("_local_scalar_dense", TORCH_FN(wrapper_DIPU___local_scalar_dense));
}

}  //end ns at
