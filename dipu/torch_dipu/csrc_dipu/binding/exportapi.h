// Copyright (c) 2023, DeepLink.
#pragma once

#include <Python.h>

#include <csrc_dipu/runtime/device/basedef.h>

namespace dipu {
DIPU_API PyMethodDef* exportTensorFunctions();
DIPU_API void exportDIPURuntime(PyObject* module);
DIPU_API void exportProfiler(PyObject* module);
}  // namespace dipu
