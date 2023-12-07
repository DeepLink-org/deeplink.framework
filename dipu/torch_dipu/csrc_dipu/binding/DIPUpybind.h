#pragma once

#include <torch/python.h>
#include <torch/types.h>  // for at::ScalarType

#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

namespace py = pybind11;

at::ScalarType dtypeToScalarType(PyObject* dtype_obj) {
  TORCH_INTERNAL_ASSERT(THPDtype_Check(dtype_obj));
  // PyTorch does not care about aliasing and is compiled with
  // `-fno-strict-aliasing`.
  // In PyTorch they would write:
  //   return reinterpret_cast<THPDtype*>(dtype_obj)->scalar_type;
  // But we do care about aliasing.
  THPDtype dtype; // NOLINT(cppcoreguidelines-pro-type-member-init)
  std::memcpy(&dtype, dtype_obj, sizeof(dtype));
  return dtype.scalar_type;
}

PyObject* scalarTypeToDtype(at::ScalarType scalar_type) {
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

  return dtype_obj;  // Note: The caller will be responsible for decreasing the
                     // refcount of dtype_obj
}

bool isDtype(PyObject* obj) { return THPDtype_Check(obj); }

template <>
struct type_caster<at::ScalarType> {
 public:
  PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));

  bool load(py::handle src, bool /*unused*/) {
    // Convert Python torch.dtype to at::ScalarType
    PyObject* obj = src.ptr();
    if (isDtype(obj)) {
      value = dtypeToScalarType(obj);
      return true;
    }
    return false;
  }

  static py::handle cast(const at::ScalarType& src,
                         py::return_value_policy /* policy */,
                         py::handle /* parent */) {
    // Convert at::ScalarType to Python torch.dtype
    return { py::handle(scalarTypeToDtype(src)) };
  }
};

}  // namespace detail
}  // namespace pybind11
