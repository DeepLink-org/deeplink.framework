// Copyright (c) 2023, DeepLink.
#pragma once

#include <pybind11/pybind11.h>
#include <csrc_dipu/common.h>

namespace dipu {
DIPU_API PyMethodDef* exportTensorFunctions();
DIPU_API void exportDIPURuntime(PyObject* module);
}