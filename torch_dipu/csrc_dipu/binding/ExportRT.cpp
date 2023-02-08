#include <c10/core/Device.h>

#include "exportapi.h"
#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/runtime/core/DIPUEvent.h>
using torch_dipu::getDIPUStreamFromPool;

namespace torch_dipu {
DIPU_API void exportDIPURuntime(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // Stream Management.
  pybind11::class_<torch_dipu::DIPUStream>(m, "_DIPUStreamBase")
    .def(py::init([]() {
          auto device_idx = devapis::current_device();
          return getDIPUStreamFromPool(device_idx);
    }))
    .def(py::init([](c10::DeviceIndex device_idx) {
          return getDIPUStreamFromPool(device_idx);
    }))
    .def("query", &torch_dipu::DIPUStream::isStreamEmpty)
    .def("synchronize",
        [](torch_dipu::DIPUStream& stream) {
          pybind11::gil_scoped_release no_gil;
          stream.synchronize();
        })
    .def("__eq__", &torch_dipu::DIPUStream::operator==);
    // .def_readwrite("device_index", &torch_dipu::DIPUStream::device_index);

  // Event
  pybind11::class_<torch_dipu::DIPUEvent>(m, "_DIPUEventBase")
    .def(pybind11::init<>())
    .def("dipu_event", [](torch_dipu::DIPUEvent& event) {
          pybind11::gil_scoped_release no_gil;
          return event.isCreated();
    })
    // .def("record", pybind11::overload_cast<const torch_dipu::DIPUEvent&>
    //                   (&torch_dipu::DIPUEvent::record), "record event on stream")
    .def("elapsed_time", &torch_dipu::DIPUEvent::elapsed_time)
    .def("synchronize",
        [](torch_dipu::DIPUEvent& event) {
          pybind11::gil_scoped_release no_gil;
          event.synchronize();
        })
    .def("query", &torch_dipu::DIPUEvent::query)
    .def("wait",
        [](torch_dipu::DIPUEvent& event, torch_dipu::DIPUStream& stream) {
          pybind11::gil_scoped_release no_gil;
          event.wait(stream);
        })
    .def_property("device", [](torch_dipu::DIPUEvent& event) {
        auto device = event.device_index();
        if (device == -1) {
          return std::string("");
        }
        std::string devicestr = "device(type='dipu', index=" + std::to_string(device) + ")";
        return devicestr;
        }, nullptr);
}
}  // end ns torch_dipu