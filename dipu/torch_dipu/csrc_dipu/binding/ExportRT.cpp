// Copyright (c) 2023, DeepLink.
#include <sstream>

#include <ATen/autocast_mode.h>
#include <c10/core/Device.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/chrono.h>

#include <csrc_dipu/aten/DIPUATenFunctions.h>
#include <csrc_dipu/base/DIPUGlobals.h>
#include <csrc_dipu/runtime/rthelper.h>
#include <csrc_dipu/utils/helpfunc.hpp>

#include "DIPUpybind.h"
#include "exportapi.h"
using dipu::DIPUEvent;
using dipu::DIPUStream;
using dipu::getDIPUStreamFromPool;
namespace py = pybind11;

namespace dipu {

static constexpr size_t kMega = 1024 * 1024;
using dipu::devapis::DIPUDeviceProperties;
using dipu::devapis::DIPUDeviceStatus;

static void registerDIPUDeviceProperties(py::module &m) {
  py::class_<DIPUDeviceProperties, std::shared_ptr<DIPUDeviceProperties>>(
      m, "_DIPUDeviceProperties")
      .def_readonly("name", &DIPUDeviceProperties::name)
      .def_readonly("major", &DIPUDeviceProperties::major)
      .def_readonly("minor", &DIPUDeviceProperties::minor)
      .def_readonly("multi_processor_count",
                    &DIPUDeviceProperties::multiProcessorCount)
      .def_readonly("total_memory", &DIPUDeviceProperties::totalGlobalMem)
      .def("__repr__", [](const DIPUDeviceProperties &prop) {
        std::ostringstream stream;
        stream << "_DIPUDeviceProperties(name='" << prop.name
               << "', major=" << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / kMega
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ")";
        return stream.str();
      });
}

static void registerDIPUDeviceStatus(py::module &m) {
  py::class_<DIPUDeviceStatus, std::shared_ptr<DIPUDeviceStatus>>(
      m, "_DIPUDeviceStatus")
      .def_readonly("free_memory", &DIPUDeviceStatus::freeGlobalMem)
      .def("__repr__", [](const DIPUDeviceStatus &status) {
        std::ostringstream stream;
        stream << "DIPUDeviceStatus(used_memory=" << status.freeGlobalMem
               << ")";
        return stream.str();
      });
}

static void exportDevices(py::module &m) {
  registerDIPUDeviceProperties(m);
  registerDIPUDeviceStatus(m);
  // Device Management.
  m.attr("dipu_vendor") = dipu::VendorTypeToStr(VENDOR_TYPE);
  m.attr("dipu_device_type") = DeviceTypeName(DIPU_DEVICE_TYPE, true);
  m.attr("dicl_backend") = DICL_BACKEND_NAME;

  m.def("_dipu_set_device", [](int idx) -> void {
    devproxy::setDevice(static_cast<devapis::deviceId_t>(idx));
  });
  m.def("_dipu_get_device_count", []() -> int {
    poison_fork();
    return devproxy::getDeviceCount();
  });
  m.def("_dipu_current_device",
        []() -> int { return static_cast<int>(devproxy::current_device()); });
  m.def("_dipu_synchronize", []() -> void {
    devproxy::syncDevice();
    return;
  });
  m.def(
      "_dipu_getDeviceProperties",
      [](int device) -> std::shared_ptr<DIPUDeviceProperties> {
        return dipu::getDevicePropertiesFromCache(device);
      },
      py::arg("device"));

  /*
    different with device properties, fill_status may cause creation of the
    device stub on the specified device, the sub will occupy mem, so caller
    should always fill status after set device() and only fill status of current
    device, otherwise you will create stub an other device.
  */
  m.def(
      "_dipu_getDeviceStatus",
      [](int device) -> std::shared_ptr<DIPUDeviceStatus> {
        return dipu::getDeviceStatus(device);
      },
      py::arg("device"));
}

static void exportStream(py::module &m) {
  // Stream Management. follow the api in torch/csrc/cuda/Stream.cpp
  pybind11::class_<DIPUStream>(m, "_DIPUStreamBase")
      .def(py::init([](int priority, c10::StreamId stream_id,
                       c10::DeviceIndex device_index, int64_t device_type,
                       uint64_t stream_ptr) {
             if (stream_id || device_index || device_type) {
               if (device_type != 0) {
                 TORCH_CHECK(static_cast<c10::DeviceType>(device_type) ==
                             dipu::DIPU_DEVICE_TYPE);
               }
               return DIPUStream(device_index, stream_id);
             } else if (stream_ptr) {
               return dipu::getStreamFromExternal(
                   reinterpret_cast<deviceStream_t>(stream_ptr),
                   devproxy::current_device());
             } else {
               return getDIPUStreamFromPool();
             }
           }),
           py::arg("priority") = 0, py::arg("stream_id") = 0,
           py::arg("device_index") = 0, py::arg("device_type") = 0,
           py::arg("stream_ptr") = 0)
      .def(py::init([](c10::DeviceIndex device_index, int isdefault) {
        return dipu::getCurrentDIPUStream(device_index);
      }))
      .def("query", &DIPUStream::isStreamEmpty)
      .def("synchronize",
           [](DIPUStream &stream) -> void {
             pybind11::gil_scoped_release no_gil;
             stream.synchronize();
           })
      .def("__eq__", &DIPUStream::operator==)
      .def("priority_range",
           // not support priority now, return a mock value.
           [](DIPUStream &stream) -> py::tuple {
             py::tuple range = pybind11::make_tuple(0, 0);
             return range;
           })
      // cpp properties
      .def_property_readonly(
          "stream_id",
          [](DIPUStream &stream) -> c10::StreamId { return stream.id(); })
      .def_property_readonly("device_index", &DIPUStream::device_index)
      .def_property_readonly(
          "device_type",
          [](DIPUStream &stream) -> int64_t {
            return static_cast<int64_t>(stream.device().type());
          })
      .def_property_readonly("dipu_stream",
                             [](DIPUStream &stream) -> uint64_t {
                               return (uint64_t)stream.rawstream();
                             })
      // use type_caster<at::Device>
      .def_property_readonly("device", [](DIPUStream &stream) -> at::Device {
        return stream.device();
      });

  m.def(
      "_dipu_setStream",
      [](c10::StreamId stream_id, c10::DeviceIndex device_index) -> void {
        dipu::setCurrentDIPUStream(DIPUStream(device_index, stream_id));
      },
      py::arg("stream_id") = 0, py::arg("device_index") = 0);

  m.def("_dipu_getCurrentStream", [](c10::DeviceIndex devIdx) -> DIPUStream {
    return dipu::getCurrentDIPUStream(devIdx);
  });
  m.def("_dipu_getDefaultStream", [](c10::DeviceIndex devIdx) -> DIPUStream {
    return dipu::getDefaultDIPUStream(devIdx);
  });
}

static void exportEvent(py::module &m) {
  // Event
  pybind11::class_<DIPUEvent>(m, "_DIPUEventBase")
      // add flag in future
      .def(py::init([](bool enable_timing, bool blocking, bool interproces) {
             return DIPUEvent();
           }),
           py::arg("enable_timing") = false, py::arg("blocking") = false,
           py::arg("interprocess") = false)
      .def("record", static_cast<void (DIPUEvent::*)()>(&DIPUEvent::record),
           "record event")
      .def("record",
           pybind11::overload_cast<const DIPUStream &>(&DIPUEvent::record),
           "record event on stream")
      .def("elapsed_time", &dipu::DIPUEvent::elapsed_time)
      .def("synchronize",
           [](DIPUEvent &self) {
             pybind11::gil_scoped_release no_gil;
             self.synchronize();
           })
      .def("query", &DIPUEvent::query)
      .def("wait",
           [](DIPUEvent &self, const DIPUStream &stream) {
             pybind11::gil_scoped_release no_gil;
             self.wait(stream);
           })

      .def_property_readonly(
          "dipu_event",
          [](DIPUEvent &self) { return (uint64_t)self.rawevent(); })
      .def_property_readonly("device", [](DIPUEvent &self) {
        auto device = self.device().value();
        return device;
      });
}

static void exportCommunicator(py::module &m) {
  pybind11::class_<ProcessGroupDICL, c10d::Backend,
                   c10::intrusive_ptr<ProcessGroupDICL>>(m, "ProcessGroupDICL")
      .def(py::init([](const c10::intrusive_ptr<c10d::Store> &store, int rank,
                       int size, const std::chrono::milliseconds &timeout) {
             return createProcessGroupDICL(store, rank, size, timeout);
           }),
           py::arg("store"), py::arg("rank"), py::arg("size"),
           py::arg("timeout") = kBackendDefaultTimeout,
           py::call_guard<py::gil_scoped_release>())
      .def("store", &ProcessGroupDICL::getStore)
      .def("timeout", [](ProcessGroupDICL &self) {
        // need enhance to support tiemout
        return kBackendDefaultTimeout;
      });

  // py::object mdist = py::module::import("torch.distributed");
  // py::object register_backend =
  // mdist.attr("Backend").attr("register_backend"); The first parameter is the
  // backend name used by user in invoking
  // torch.distributed.init_process_group().
  // register_backend(dipu::DICL_BACKEND_NAME,
  // py::cpp_function(createProcessGroupDICL));
}

static void exportMemCaching(py::module &m) {
  m.def("_dipu_emptyCache", []() { emptyCachedMem(); });

  m.def("init_resource", []() { initResource(); });

  m.def("release_all_resources", []() { releaseAllResources(); });

  m.def("memory_reserved", [](const c10::Device &device) -> size_t {
    return memoryReserved(device);
  });

  m.def("memory_allocated", [](const c10::Device &device) -> size_t {
    return memoryAllocated(device);
  });

  m.def("max_memory_reserved", [](const c10::Device &device) -> size_t {
    return maxMemoryReserved(device);
  });

  m.def("max_memory_allocated", [](const c10::Device &device) -> size_t {
    return maxMemoryAllocated(device);
  });
}

static void patchStorage(py::module &m) {
  // incremental patch StorageMethods.cpp THPStorage_resize_()
  m.def("storage_resize_",
        [](at::Storage stor, int64_t newsize) -> at::Storage {
          if (stor.device_type() != DIPU_DEVICE_TYPE) {
            TORCH_CHECK(false,
                        "UntypedStorage.resize_: dipu storage resize not "
                        "support other device type ",
                        stor.device_type());
          } else {
            dipu::native::DIPUATenFunctions::resize_bytes_dipu(
                stor.unsafeGetStorageImpl(), newsize);
            return stor;
          }
        });
}

static void patchTensor(py::module &m) {
  m.def("is_dipu",
        [](at::Tensor self) -> bool { return dipu::isDeviceTensor(self); });
}

static void exportGenerator(py::module &m) {
  m.def("_manual_seed",
        [](at::DeviceIndex idx, uint64_t seed) { manual_seed(idx, seed); });

  m.def("_seed", [](at::DeviceIndex idx) { seed(idx); });

  m.def("_initial_seed",
        [](at::DeviceIndex idx) -> uint64_t { return initial_seed(idx); });

  m.def("_get_rng_state",
        [](at::DeviceIndex idx) -> at::Tensor { return get_rng_state(idx); });

  m.def("_set_rng_state", [](at::DeviceIndex idx, at::Tensor state) {
    set_rng_state(idx, state);
  });

  m.def("_is_in_bad_fork", []() -> bool { return is_in_bad_fork(); });

  m.def("_create_dipu_generator", [](int idx) -> at::Generator {
    at::DeviceIndex index = static_cast<at::DeviceIndex>(idx);
    return createDIPUGenerator(index);
  });
}

static void exportAutocast(py::module &m) {
  m.def("get_autocast_dipu_dtype", []() -> at::ScalarType {
    return at::autocast::get_autocast_xpu_dtype();
  });
  m.def("is_autocast_dipu_enabled",
        []() -> bool { return at::autocast::is_xpu_enabled(); });
  m.def("set_autocast_dipu_enabled",
        [](bool enabled) { at::autocast::set_xpu_enabled(enabled); });
  m.def("set_autocast_dipu_dtype", [](at::ScalarType dtype) {
    at::autocast::set_autocast_xpu_dtype(dtype);
  });
}

extern void patchTorchCsrcDevice(PyObject *module);

DIPU_API void exportDIPURuntime(PyObject *module) {
  auto m = py::handle(module).cast<py::module>();
  patchTorchCsrcDevice(module);
  exportDevices(m);
  exportStream(m);
  exportEvent(m);
  exportCommunicator(m);
  exportMemCaching(m);
  patchStorage(m);
  patchTensor(m);
  exportGenerator(m);
  exportAutocast(m);
}
}  // namespace dipu