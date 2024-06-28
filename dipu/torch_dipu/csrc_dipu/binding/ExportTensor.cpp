// Copyright (c) 2023, DeepLink.
#include <c10/core/Backend.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_types.h>

#include <Python.h>
#include <object.h>

#include <csrc_dipu/base/basedef.h>

#include "exportapi.h"

using std::unique_ptr;

namespace dipu {
static at::Tensor dispatch_to(
    const at::Tensor& self, at::Device device, bool non_blocking, bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing.
  // However, the behavior of aten::to is different with respect to
  // TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they
  // should be populated with the default values (eg. float for scalar type). By
  // explicitly copying over the tensor options here we fully specify all tensor
  // options and thus record the proper trace
  return self.to(
      self.options().device(device).memory_format(optional_memory_format),
      non_blocking, copy);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static auto splitArgs(PyObject* args) {
  ssize_t rawSize = PyTuple_Size(args);
  PyObject* newArgsTuple = PyTuple_New(rawSize - 1);
  for (int i = 1; i < rawSize; i++) {
    auto arg = PyTuple_GET_ITEM(args, i);
    Py_INCREF(arg);
    PyTuple_SetItem(newArgsTuple, i - 1, arg);
  }

  using PyObjectUniquePointer = std::unique_ptr<PyObject, void (*)(PyObject*)>;
  auto self =
      PyObjectUniquePointer(PyTuple_GET_ITEM(args, 0), [](PyObject* p) {});
  auto newArgs =
      PyObjectUniquePointer(newArgsTuple, [](PyObject* p) { Py_DecRef(p); });
  return std::make_tuple(std::move(self), std::move(newArgs));
}

// first parameter is export module torchdipu_module, not self tensor
static PyObject* THPVariable_dipu(PyObject* module, PyObject* args,
                                  PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"dipu(Device? device=None, bool non_blocking=False, *, MemoryFormat? "
       "memory_format=None)",
       "dipu(Device? device=None, bool async=False, *, MemoryFormat? "
       "memory_format=None)|deprecated"});

  auto res = splitArgs(args);
  PyObject* self = std::get<0>(res).get();
  PyObject* newArgs = std::get<1>(res).get();

  auto& self_ = THPVariable_Unpack(self);
  torch::ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, newArgs, kwargs, parsed_args);

  if (r.has_torch_function()) {
    return torch::handle_torch_function(r, self, newArgs, kwargs,
                                        THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(dipu::DIPU_DEVICE_TYPE) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.type() == dipu::DIPU_DEVICE_TYPE,
              "Invalid device, must be dipu device");
  return THPVariable_Wrap(
      dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

// we prefer to use pybind11 to export patch func, cpython is used only patching
// tensor-func which has complex dynamic parameters not easy to parsed by
// pybind.
DIPU_API PyMethodDef* exportTensorFunctions() {
  static std::array<PyMethodDef, 2> TorchTensorMethods = {
      {{"dipu", castPyCFunctionWithKeywords(THPVariable_dipu),
        METH_VARARGS | METH_KEYWORDS, nullptr},
       {nullptr, nullptr, 0, nullptr}}};
  return TorchTensorMethods.data();
}

// PyTensorType was defined in torch/csrc/tensor/python_tensor.cpp only, we can
// only copy it to here to use and follow the original definition.
struct PyTensorType {
  PyTypeObject py_type;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_cuda;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
  char name[64];
  int backend;
  int scalar_type;

  at::Backend get_backend() const { return static_cast<at::Backend>(backend); }

  c10::DispatchKey get_dispatch_key() const {
    return c10::backendToDispatchKey(static_cast<at::Backend>(backend));
  }

  at::ScalarType get_scalar_type() const {
    return static_cast<at::ScalarType>(scalar_type);
  }
};

static_assert(std::is_standard_layout<PyTensorType>::value,
              "PyTensorType must be standard layout");

// torch's Tensor_new checks torch::utils::cuda_enabled() for cuda tensor, we
// need to get rid of this check.
static PyObject* mock_Tensor_new(PyTypeObject* type, PyObject* args,
                                 PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *(reinterpret_cast<PyTensorType*>(type));
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(
      tensor_type.get_dispatch_key(), tensor_type.get_scalar_type(), args,
      kwargs));
  END_HANDLE_TH_ERRORS
}

static inline at::Backend dipu_mock_backend(at::Backend backend) {
  switch (backend) {
    case at::Backend::CUDA:
      return DIPU_BACKEND_TYPE;
    case at::Backend::SparseCUDA:
      return DIPU_BACKEND_SPARSE_TYPE;
    default:
      return backend;
  }
}

// see test: unittests/test_mock_cudatensor.py test_cuda_tensor_type for detail
static void mockCudaTensorType() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) {
    throw python_error();
  }

  auto tensor_classes = THPObjectPtr(
      PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) {
    throw python_error();
  }

  auto tensor_cls_seq = THPObjectPtr(PySequence_Fast(
      tensor_classes, "torch._tensor_classes has been modified\n"));
  if (!tensor_cls_seq) {
    throw python_error();
  }

  Py_ssize_t len = PySequence_Fast_GET_SIZE(tensor_cls_seq.get());
  PyObject** tensor_type_array = PySequence_Fast_ITEMS(tensor_cls_seq.get());

  for (Py_ssize_t i = 0; i < len; ++i) {
    // assume no one change the items in torch._tensor_classes, i.e. assume
    // they can be reinterpreted as PyTensorType.
    // NOLINTNEXTLINE(modernize-use-auto)
    PyTensorType* tensor_type =
        reinterpret_cast<PyTensorType*>(tensor_type_array[i]);
    tensor_type->py_type.tp_new = mock_Tensor_new;
    tensor_type->backend =
        static_cast<int>(dipu_mock_backend(tensor_type->get_backend()));
  }
}

void patchTorchTensor(py::module& m) {
  m.def("_mockCudaTensorType", mockCudaTensorType);
}

}  // namespace dipu
