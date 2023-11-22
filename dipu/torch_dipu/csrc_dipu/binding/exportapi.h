// Copyright (c) 2023, DeepLink.
#pragma once

#include <pybind11/pybind11.h>

#include <csrc_dipu/base/basedef.h>

namespace dipu {
DIPU_API PyMethodDef* exportTensorFunctions();
DIPU_API void exportDIPURuntime(PyObject* module);
DIPU_API void exportProfiler(PyObject* module);
}  // namespace dipu
