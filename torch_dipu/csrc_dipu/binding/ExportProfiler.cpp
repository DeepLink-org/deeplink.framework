
// Copyright (c) 2023, DeepLink.
#include <csrc_dipu/profiler/profiler.h>
#include <pybind11/chrono.h>
#include <torch/csrc/utils/pybind.h>

#include "exportapi.h"

namespace py = pybind11;

namespace dipu {

void exportProfiler(PyObject* module) {
    auto m = py::handle(module).cast<py::module>();

    m.def("profile_start", &dipu::profile::startProfile);
    m.def("profile_end", &dipu::profile::endProfile);
    m.def("profiler_flush", &dipu::profile::FlushAllRecords);
    py::class_<dipu::profile::Record>(m, "_DIPUProfilerRecord")
        .def_readonly("name", &dipu::profile::Record::name)
        .def_readonly("opid", &dipu::profile::Record::opId)
        .def_readonly("begin", &dipu::profile::Record::begin)
        .def_readonly("end", &dipu::profile::Record::end)
        .def_readonly("thread_idx", &dipu::profile::Record::threadIdx);
    m.def("get_record", &dipu::profile::getRecordList);
}

}  // namespace dipu