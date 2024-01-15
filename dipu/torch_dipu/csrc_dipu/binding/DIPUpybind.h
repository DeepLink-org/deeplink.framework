#pragma once

#include <torch/python.h>
#include <torch/types.h>  // for at::ScalarType

#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

template <>
struct type_caster<at::ScalarType> {
 public:
  PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));

  // See: https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  bool load(handle src, bool /* unused */) {
    // Convert Python torch.dtype to at::ScalarType
    auto source = src.ptr();
    if (not THPDtype_Check(source)) {
      return false;
    }
    value = dtypeToScalarType(source);
    return true;
  }

  static handle cast(at::ScalarType src, return_value_policy /* policy */,
                     handle /* parent */) {
    // Convert at::ScalarType to Python torch.dtype
    return {scalarTypeToDtype(src)};
  }

 private:
  static at::ScalarType dtypeToScalarType(PyObject* dtype_obj) {
    TORCH_INTERNAL_ASSERT(THPDtype_Check(dtype_obj));
    // PyTorch does not care about aliasing and is compiled with
    // `-fno-strict-aliasing`.
    // In PyTorch they would write:
    //   return reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;
    // But we do care about aliasing.
    THPDtype dtype{};
    std::memcpy(&dtype, dtype_obj, sizeof(dtype));
    return dtype.scalar_type;
  }

  static PyObject* scalarTypeToDtype(at::ScalarType scalar_type) {
    const char* dtype_name = nullptr;
    switch (scalar_type) {
      case at::ScalarType::Float:
        dtype_name = "float32";
        break;
      case at::ScalarType::Double:
        dtype_name = "float64";
        break;
      case at::kHalf:
        dtype_name = "float16";
        break;
      case at::kBFloat16:
        dtype_name = "bfloat16";
        break;
      // ... handle other scalar types here
      default:
        throw std::runtime_error("Unsupported scalar type");
    }

    PyObject* torch_module = PyImport_ImportModule("torch");
    TORCH_INTERNAL_ASSERT(torch_module);

    PyObject* dtype_obj = PyObject_GetAttrString(torch_module, dtype_name);
    TORCH_INTERNAL_ASSERT(dtype_obj);

    Py_DECREF(torch_module);  // Decrement the refcount for the torch module

    return dtype_obj;  // Note: The caller will be responsible for decreasing
                       // the refcount of dtype_obj
  }
};

}  // namespace detail
}  // namespace pybind11
