
// Copyright (c) 2023, DeepLink.

#include <set>
#include <unordered_set>

#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/chrono.h>

#include "csrc_dipu/profiler/profiler.h"
#include "csrc_dipu/profiler/profiler_kineto.h"
#include "csrc_dipu/profiler/profiler_python.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

#include "exportapi.h"

namespace py = pybind11;

namespace dipu {

void exportProfiler(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  m.def("_prepare_profiler", profile::prepareProfiler);
  m.def("_enable_profiler", profile::enableProfiler, py::arg("config"),
        py::arg("activities"),
        py::arg("scopes") = std::unordered_set<at::RecordScope>());
  m.def("_disable_profiler", profile::disableProfiler);
  m.def("_add_metadata_json", profile::addMetadataJson);
  m.def("_kineto_step", profile::profilerStep);
  m.def("_supported_activities", []() {
    std::set<torch::profiler::impl::ActivityType> activities{
        torch::profiler::impl::ActivityType::CPU};
    if (devproxy::getDeviceCount() > 0) {
      activities.insert(torch::profiler::impl::ActivityType::CUDA);
    }
    return activities;
  });
  profile::init();
}

}  // namespace dipu
