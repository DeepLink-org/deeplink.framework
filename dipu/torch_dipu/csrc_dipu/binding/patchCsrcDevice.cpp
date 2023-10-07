// Copyright (c) 2023, DeepLink.

#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>

#include <ATen/Device.h>
#include <c10/util/Exception.h>
#include <torch/csrc/Device.h>

#include <structmember.h>
#include <cstring>
#include <limits>
#include <sstream>

#include "exportapi.h"

namespace dipu {

static bool python_device_as_cuda = false;

static at::DeviceType _get_dipu_python_type(const at::Device& device) {
  if (device.type() == DIPU_DEVICE_TYPE && python_device_as_cuda) {
    return at::DeviceType::CUDA;
  }
  return device.type();
}

PyObject* _THPDevice_type(THPDevice* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  std::ostringstream oss;
  oss << _get_dipu_python_type(self->device);
  return THPUtils_packString(oss.str().c_str());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* _THPDevice_index(THPDevice* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (self->device.has_index()) {
    return THPUtils_packInt64(self->device.index());
  } else {
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* DIPU_THPDevice_repr(THPDevice* self) {
  std::ostringstream oss;
  oss << "device(type=\'" << _get_dipu_python_type(self->device) << "\'";
  if (self->device.has_index()) {
    // `self->device.index()` returns uint8_t which is treated as ascii while
    // printing, hence casting it to uint16_t.
    // https://stackoverflow.com/questions/19562103/uint8-t-cant-be-printed-with-cout
    oss << ", index=" << static_cast<uint16_t>(self->device.index());
  }
  oss << ")";
  return THPUtils_packString(oss.str().c_str());
}


PyObject* DIPU_THPDevice_str(THPDevice* self) {
  std::ostringstream oss;
  oss << _get_dipu_python_type(self->device);
  return THPUtils_packString(oss.str().c_str());
}

static struct PyGetSetDef DIPU_THPDevice_properties[] = {
    {"type", (getter)_THPDevice_type, nullptr, nullptr, nullptr},
    {"index", (getter)_THPDevice_index, nullptr, nullptr, nullptr},
    {nullptr}};

void patchTorchCsrcDevice(PyObject* module) {
  // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_dict
  THPDeviceType.tp_dict = nullptr;
  // change Type properties
  THPDeviceType.tp_getset = DIPU_THPDevice_properties;
  THPDeviceType.tp_repr = (reprfunc)DIPU_THPDevice_repr;
  THPDeviceType.tp_str = (reprfunc)DIPU_THPDevice_str;

  // change THPDeviceType as an overriable class need add some other prperties in PyTypeObject,
  // It may cause problems and seem un-necessary, so we keep the THPDeviceType as immutable. 
  THPDeviceType.tp_flags = Py_TPFLAGS_DEFAULT;   // | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;

  if (PyType_Ready(&THPDeviceType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPDeviceType);

  auto m = py::handle(module).cast<py::module>();

  m.def("_get_python_device_as_cuda", []() -> bool {
    return python_device_as_cuda;
  });

  m.def ("_set_python_device_as_cuda", [](bool as_cuda) -> void {
    python_device_as_cuda = as_cuda;
  });

  // not really 'export' new type but change original THPDeviceType is enough
  // if (PyModule_AddObject(module, "device", (PyObject*)&THPDeviceType) != 0) {
  //   throw python_error();
  // }
}
}  // namespace dipu