#include <c10/core/Device.h>

#include "exportapi.h"
#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/runtime/core/DIPUEvent.h>
using dipu::getDIPUStreamFromPool;

namespace py = pybind11;

namespace dipu {

static void exportDevices(py::module& m) {
   // Device Management.
  m.def("_dipu_set_device", [](int idx) -> void { 
    devapis::setDevice(static_cast<devapis::deviceId_t>(idx)); 
  });
  m.def("_dipu_get_device_count", []() -> int { 
    return devapis::getDeviceCount();
  });
  m.def("_dipu_current_device", []() -> int {
    return static_cast<int>(devapis::current_device()); 
  });
  m.def("_dipu_synchronize", []() -> void { 
    devapis::syncDevice(); 
    return;
  });
}

DIPU_API void exportDIPURuntime(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  exportDevices(m);

  // Stream Management.
  pybind11::class_<dipu::DIPUStream>(m, "_DIPUStreamBase")
    .def(py::init([]() {
          auto device_idx = devapis::current_device();
          return getDIPUStreamFromPool(device_idx);
    }))
    .def(py::init([](c10::DeviceIndex device_idx) {
          return getDIPUStreamFromPool(device_idx);
    }))
    .def("query", &dipu::DIPUStream::isStreamEmpty)
    .def("synchronize",
        [](dipu::DIPUStream& stream) {
          pybind11::gil_scoped_release no_gil;
          stream.synchronize();
        })
    .def("__eq__", &dipu::DIPUStream::operator==);
    // .def_readwrite("device_index", &dipu::DIPUStream::device_index);

  // Event
  pybind11::class_<dipu::DIPUEvent>(m, "_DIPUEventBase")
    .def(pybind11::init<>())
    .def("dipu_event", [](dipu::DIPUEvent& event) {
          pybind11::gil_scoped_release no_gil;
          return event.isCreated();
    })
    // .def("record", pybind11::overload_cast<const dipu::DIPUEvent&>
    //                   (&dipu::DIPUEvent::record), "record event on stream")
    .def("elapsed_time", &dipu::DIPUEvent::elapsed_time)
    .def("synchronize",
        [](dipu::DIPUEvent& event) {
          pybind11::gil_scoped_release no_gil;
          event.synchronize();
        })
    .def("query", &dipu::DIPUEvent::query)
    .def("wait",
        [](dipu::DIPUEvent& event, dipu::DIPUStream& stream) {
          pybind11::gil_scoped_release no_gil;
          event.wait(stream);
        })
    .def_property("device", [](dipu::DIPUEvent& event) {
        auto device = event.device_index();
        if (device == -1) {
          return std::string("");
        }
        std::string devicestr = "device(type='dipu', index=" + std::to_string(device) + ")";
        return devicestr;
        }, nullptr);
}
}  // end ns dipu