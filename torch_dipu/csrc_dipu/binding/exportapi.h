#pragma once

#include <torch/csrc/jit/python/pybind.h>


#define DIPU_API __attribute__ ((visibility ("default")))

#define DIPU_WEAK  __attribute__((weak))

// "default", "hidden", "protected" or "internal
#define DIPU_HIDDEN __attribute__ ((visibility ("hidden")))

namespace torch_dipu {
DIPU_API PyMethodDef* exportTensorFunctions();
DIPU_API void exportDIPURuntime(PyObject* module);
}