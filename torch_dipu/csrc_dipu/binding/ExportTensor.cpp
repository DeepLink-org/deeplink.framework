#include <torch/csrc/utils/tensor_types.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include "exportapi.h"
#include <csrc_dipu/common.h>

namespace dipu {
static at::Tensor dispatch_to(const at::Tensor & self, c10::Device device, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing. However, the behavior of aten::to
  // is different with respect to TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they should be populated with the
  // default values (eg. float for scalar type). By explicitly copying over the tensor options here we fully
  // specify all tensor options and thus record the proper trace
  return self.to(self.options().device(device).memory_format(optional_memory_format), non_blocking, copy);
}

static at::Tensor dispatch_to(const at::Tensor & self, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AutoNoGIL no_gil;
  return self.to(self.options().memory_format(optional_memory_format), non_blocking, copy);
}

static at::Tensor dispatch_to(const at::Tensor & self, c10::ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

static at::Tensor dispatch_to(const at::Tensor & self, c10::Device device, c10::ScalarType dtype, bool non_blocking, bool copy, c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

static const char* _backend_to_string_dipu(const at::Backend& backend) {
  switch (backend) {
    case at::Backend::CPU: return "torch";
    case dipu::DIPU_Backend_TYPE: return "torch.dipu";
    default: AT_ERROR("Unimplemented backend ", backend);
  }
}

std::string _options_to_string_dipu(const at::TensorOptions options) {
  std::ostringstream ss;
  ss << _backend_to_string_dipu(options.backend()) << "." << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return ss.str();
}

static PyObject* THPVariable_dipu(PyObject* self, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "dipu(Tensor temp, Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "dipu(Tensor temp, Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  auto local_device = r.isNone(1) ? c10::Device(dipu::DIPU_DEVICE_TYPE) : r.device(1);
  auto device = c10::Device(dipu::DIPU_DEVICE_TYPE, local_device.index());
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK((device.type() == dipu::DIPU_DEVICE_TYPE), "Invalid device, must be npu device");
  return THPVariable_Wrap(dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_type(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "type(Tensor temp, PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
    "type(Tensor temp, PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"
  });
  
  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if(r.has_torch_function()) {
    return torch::handle_torch_function(r, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.isNone(1)) {
    return THPUtils_packString(_options_to_string_dipu(self_.options()));
  }
  auto obj = r.pyobject(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  std::string type_name;
  bool is_dtype = false;
  if (PyType_Check(obj)) {
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    is_dtype = true;
  } else {
    throw torch::TypeError("dtype must be a type, str, or dtype object");
  }
  c10::ScalarType scalar_type;
  c10::Device device = self_.device();
  if (is_dtype) {
    scalar_type = r.scalartype(1);
  } else {
    at::TensorOptions options = torch::utils::options_from_string(type_name);
    scalar_type = at::typeMetaToScalarType(options.dtype());
    auto device_type = options.device().type();
    if (device_type != device.type()) {
      device = at::Device(device_type);
    }
  };
  return THPVariable_Wrap(dispatch_to(self_, device, scalar_type, r.toBool(1), false, opt_memory_format));
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
  {"type", castPyCFunctionWithKeywords(THPVariable_type), METH_VARARGS | METH_KEYWORDS, NULL},
  {"dipu", castPyCFunctionWithKeywords(THPVariable_dipu), METH_VARARGS | METH_KEYWORDS, NULL},
  {nullptr, nullptr, 0, nullptr}
};

DIPU_API PyMethodDef* exportTensorFunctions() {
  return TorchTensorMethods;
}
} // end ns dipu