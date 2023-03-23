#include <torch/csrc/utils/tensor_types.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include "exportapi.h"
#include <csrc_dipu/common.h>

namespace dipu {
static at::Tensor dispatch_to(const at::Tensor& self, at::Device device, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

static std::shared_ptr<PyObject*[2]> splitArgs(PyObject* args) {
  ssize_t rawSize = PyTuple_Size(args);
  PyObject* newArgs = PyTuple_New(rawSize - 1);
  std::shared_ptr<PyObject*[2]> result(new PyObject*[2], [](PyObject** p){
      // if (p[1]) {    // cause segfault, why?
      //   Py_DECREF(p[1]);
      // } 
      delete[] p;
      p = nullptr;
    });
  // 0 is self
  result[0] = PyTuple_GET_ITEM(args, 0);
  result[1] = newArgs;

  for (int i = 1; i < rawSize; i++) {
    auto arg = PyTuple_GET_ITEM(args, i);
    PyTuple_SetItem(newArgs, i-1, arg);
  }
  return result;
}

// first parameter is export module torchdipu_module, not self tensor
static PyObject* THPVariable_dipu(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static  torch::PythonArgParser parser({
    "dipu(Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "dipu(Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });

  auto res = splitArgs(args);
  PyObject* self = res[0];
  PyObject* newArgs = res[1];

  auto& self_ = THPVariable_Unpack(self);
  torch::ParsedArgs<3> parsed_args;
  auto r = parser.parse(self, newArgs, kwargs, parsed_args);

  if(r.has_torch_function()) {
    return torch::handle_torch_function(r, self, newArgs, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device = r.isNone(0) ? at::Device(dipu::DIPU_DEVICE_TYPE) : r.device(0);
  auto opt_memory_format = r.memoryformatOptional(2);
  TORCH_CHECK(device.type() == dipu::DIPU_DEVICE_TYPE, "Invalid device, must be dipu device");
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(1), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_dipu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "is_dipu(Tensor temp)"
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  return torch::autograd::utils::wrap(dipu::isDeviceTensor(self_));
  END_HANDLE_TH_ERRORS
}

static PyMethodDef TorchTensorMethods[] = {
  {"is_dipu", castPyCFunctionWithKeywords(THPVariable_is_dipu), METH_VARARGS | METH_KEYWORDS, NULL},
  // {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, NULL},
  {"dipu", castPyCFunctionWithKeywords(THPVariable_dipu), METH_VARARGS | METH_KEYWORDS, NULL},
  {nullptr, nullptr, 0, nullptr}
};

DIPU_API PyMethodDef* exportTensorFunctions() {
  return TorchTensorMethods;
}
} // end ns dipu