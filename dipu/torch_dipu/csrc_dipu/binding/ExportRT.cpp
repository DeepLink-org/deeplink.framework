// Copyright (c) 2023, DeepLink.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/autocast_mode.h>
#include <ATen/core/ATen_fwd.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/utils/pybind.h>

#include <object.h>
#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/base/DIPUGlobals.h"
#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/metrics/metrics.h"
#include "csrc_dipu/runtime/core/DIPUEvent.h"
#include "csrc_dipu/runtime/core/DIPUGeneratorImpl.h"
#include "csrc_dipu/runtime/core/DIPUStream.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocatorUtils.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingDeviceAllocator.h"
#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/runtime/device/deviceapis.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include "csrc_dipu/runtime/distributed/ProcessGroupDICL.h"
#include "csrc_dipu/runtime/rthelper.h"
#include "csrc_dipu/utils/helpfunc.hpp"
#include "csrc_dipu/utils/vender_helper.hpp"
#include "csrc_dipu/vendor/vendorapi.h"

#include "DIPUpybind.h"  // IWYU pragma: keep
#include "exportapi.h"

namespace py = pybind11;

namespace dipu {

namespace {

constexpr auto kMega = static_cast<size_t>(1024 * 1024);

void registerDIPUDeviceProperties(py::module& m) {
  py::class_<DIPUDeviceProperties, std::shared_ptr<DIPUDeviceProperties>>(
      m, "_DIPUDeviceProperties")
      .def_readonly("name", &DIPUDeviceProperties::name)
      .def_readonly("major", &DIPUDeviceProperties::major)
      .def_readonly("minor", &DIPUDeviceProperties::minor)
      .def_readonly("multi_processor_count",
                    &DIPUDeviceProperties::multiProcessorCount)
      .def_readonly("total_memory", &DIPUDeviceProperties::totalGlobalMem)
      .def("__repr__", [](const DIPUDeviceProperties& prop) {
        std::ostringstream stream;
        stream << "_DIPUDeviceProperties(name='" << prop.name
               << "', major=" << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / kMega
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ")";
        return stream.str();
      });
}

void registerDIPUDeviceStatus(py::module& m) {
  py::class_<DIPUDeviceStatus, std::shared_ptr<DIPUDeviceStatus>>(
      m, "_DIPUDeviceStatus")
      .def_readonly("free_memory", &DIPUDeviceStatus::freeGlobalMem)
      .def_readonly("total_memory", &DIPUDeviceStatus::totalGlobalMem)
      .def("__repr__", [](const DIPUDeviceStatus& status) {
        std::ostringstream stream;
        stream << "DIPUDeviceStatus(free_memory="
               << status.freeGlobalMem / kMega
               << "MB, total_memory=" << status.totalGlobalMem / kMega << "MB)";
        return stream.str();
      });
}

void exportDevices(py::module& m) {
  registerDIPUDeviceProperties(m);
  registerDIPUDeviceStatus(m);
  // Device Management.
  // dipu_vendor should be dipu_vendor_device, but we keep it for compatibility
  m.attr("dipu_vendor") = dipu::VendorDeviceTypeToStr(kDipuVendorDeviceType);
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
        []() { return static_cast<int>(devproxy::current_device()); });
  m.def("_dipu_synchronize", devproxy::syncDevice);
  m.def("_dipu_getDeviceProperties", getDevicePropertiesFromCache,
        py::arg("device"));

  /*
    different with device properties, fill_status may cause creation of the
    device stub on the specified device, the sub will occupy mem, so caller
    should always fill status after set device() and only fill status of current
    device, otherwise you will create stub an other device.
  */
  m.def("_dipu_getDeviceStatus", getDeviceStatus, py::arg("device"));
}

void exportStream(py::module& m) {
  // Stream Management. follow the api in torch/csrc/cuda/Stream.cpp
  py::class_<DIPUStream>(m, "_DIPUStreamBase")
      .def(py::init([](int priority, c10::StreamId stream_id,
                       c10::DeviceIndex device_index, int64_t device_type,
                       uint64_t stream_ptr) {
             if (stream_id || device_index || device_type) {
               if (device_type != 0) {
                 TORCH_CHECK(static_cast<c10::DeviceType>(device_type) ==
                             dipu::DIPU_DEVICE_TYPE);
               }
               return DIPUStream(device_index, stream_id);
             }
             if (stream_ptr) {
               return dipu::getStreamFromExternal(
                   // NOLINTNEXTLINE(performance-no-int-to-ptr)
                   reinterpret_cast<deviceStream_t>(stream_ptr),
                   devproxy::current_device());
             }
             return getDIPUStreamFromPool();
           }),
           py::arg("priority") = 0, py::arg("stream_id") = 0,
           py::arg("device_index") = 0, py::arg("device_type") = 0,
           py::arg("stream_ptr") = 0)
      .def(py::init([](c10::DeviceIndex device_index, int isdefault) {
        return dipu::getCurrentDIPUStream(device_index);
      }))
      .def("query", &DIPUStream::isStreamEmpty)
      .def("synchronize",
           [](DIPUStream& stream) -> void {
             py::gil_scoped_release no_gil;
             stream.synchronize();
           })
      .def("__eq__", &DIPUStream::operator==)
      .def("priority_range",
           // not support priority now, return a mock value.
           [](DIPUStream& stream) -> py::tuple {
             py::tuple range = py::make_tuple(0, 0);
             return range;
           })
      // cpp properties
      .def_property_readonly(
          "stream_id",
          [](DIPUStream& stream) -> c10::StreamId { return stream.id(); })
      .def_property_readonly("device_index", &DIPUStream::device_index)
      .def_property_readonly(
          "device_type",
          [](DIPUStream& stream) -> int64_t {
            return static_cast<int64_t>(stream.device().type());
          })
      .def_property_readonly(
          "dipu_stream",
          [](DIPUStream& stream) -> uint64_t {
            return reinterpret_cast<uint64_t>(stream.rawstream());
          })
      // use type_caster<at::Device>
      .def_property_readonly("device", [](DIPUStream& stream) -> at::Device {
        return stream.device();
      });

  m.def(
      "_dipu_setStream",
      [](c10::StreamId stream_id, c10::DeviceIndex device_index) -> void {
        dipu::setCurrentDIPUStream(DIPUStream(device_index, stream_id));
      },
      py::arg("stream_id") = 0, py::arg("device_index") = 0);

  m.def("_dipu_getCurrentStream", getCurrentDIPUStream);
  m.def("_dipu_getDefaultStream", getDefaultDIPUStream);
}

void exportEvent(py::module& m) {
  // Event
  py::class_<DIPUEvent>(m, "_DIPUEventBase")
      // add flag in future
      .def(py::init([](bool enable_timing, bool blocking, bool interproces) {
             return DIPUEvent();
           }),
           py::arg("enable_timing") = false, py::arg("blocking") = false,
           py::arg("interprocess") = false)
      .def("record", py::overload_cast<>(&DIPUEvent::record), "record event")
      .def("record", py::overload_cast<const DIPUStream&>(&DIPUEvent::record),
           "record event on stream")
      .def("elapsed_time", &dipu::DIPUEvent::elapsed_time)
      .def("synchronize",
           [](DIPUEvent& self) {
             py::gil_scoped_release no_gil;
             self.synchronize();
           })
      .def("query", &DIPUEvent::query)
      .def("wait",
           [](DIPUEvent& self, const DIPUStream& stream) {
             py::gil_scoped_release no_gil;
             self.wait(stream);
           })

      .def_property_readonly(
          "dipu_event",
          [](DIPUEvent& self) {
            return reinterpret_cast<uint64_t>(self.rawevent());
          })
      .def_property_readonly(
          "device", [](DIPUEvent& self) { return self.device().value(); });
}

void exportCommunicator(py::module& m) {
  py::class_<ProcessGroupDICL, c10d::Backend,
             c10::intrusive_ptr<ProcessGroupDICL>>(m, "ProcessGroupDICL")
      .def(py::init([](const c10::intrusive_ptr<c10d::Store>& store, int rank,
                       int size, const std::chrono::milliseconds& timeout) {
             return createProcessGroupDICL(store, rank, size, timeout);
           }),
           py::arg("store"), py::arg("rank"), py::arg("size"),
           py::arg("timeout") = kBackendDefaultTimeout,
           py::call_guard<py::gil_scoped_release>())
      .def("store", &ProcessGroupDICL::getStore)
      .def("get_comm_name",
           [](ProcessGroupDICL& self, const at::DeviceIndex device_index) {
             return std::string(self.getCommName(device_index));
           })
      .def("timeout", [](ProcessGroupDICL& self) {
        // need enhance to support timeout
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

void exportMemCaching(py::module& m) {
  m.def("_dipu_emptyCache", emptyCachedMem);
  m.def("init_resource", initResource);
  m.def("release_all_resources", releaseAllResources);
  m.def("memory_reserved", memoryReserved);
  m.def("memory_allocated", memoryAllocated);
  m.def("max_memory_reserved", maxMemoryReserved);
  m.def("max_memory_allocated", maxMemoryAllocated);
  m.def("reset_peak_memory_stats", resetPeakStats);
  m.def("_dipu_dipuCachingAllocator_set_allocator_settings",
        dipu::allocator::setAllocatorSettings);
}

void patchStorage(py::module& m) {
  // incremental patch StorageMethods.cpp THPStorage_resize_()
  m.def("storage_resize_",
        [](at::Storage stor, int64_t newsize) -> at::Storage {
          if (stor.device_type() != DIPU_DEVICE_TYPE) {
            TORCH_CHECK(false,
                        "UntypedStorage.resize_: dipu storage resize not "
                        "support other device type ",
                        stor.device_type());
          }
          dipu::native::dipu_aten::resize_bytes_dipu(
              stor.unsafeGetStorageImpl(), newsize);
          return stor;
        });
}

void patchTensor(py::module& m) { m.def("is_dipu", isDeviceTensor); }

void exportNativeMemoryFormat(py::module& m) {
  py::enum_<NativeMemoryFormat_t> formats =
      py::enum_<NativeMemoryFormat_t>(m, "NativeMemoryFormat");
#if DIPU_VENDOR_NAME_ASCEND
  formats.value("UNDEFINED", NativeMemoryFormat_t::UNDEFINED)
      .value("NCHW", NativeMemoryFormat_t::NCHW)
      .value("NHWC", NativeMemoryFormat_t::NHWC)
      .value("ND", NativeMemoryFormat_t::ND)
      .value("NC1HWC0", NativeMemoryFormat_t::NC1HWC0)
      .value("FRACTAL_Z", NativeMemoryFormat_t::FRACTAL_Z)
      .value("NC1HWC0_C04", NativeMemoryFormat_t::NC1HWC0_C04)
      .value("HWCN", NativeMemoryFormat_t::HWCN)
      .value("NDHWC", NativeMemoryFormat_t::NDHWC)
      .value("FRACTAL_NZ", NativeMemoryFormat_t::FRACTAL_NZ)
      .value("NCDHW", NativeMemoryFormat_t::NCDHW)
      .value("NDC1HWC0", NativeMemoryFormat_t::NDC1HWC0)
      .value("FRACTAL_Z_3D", NativeMemoryFormat_t::FRACTAL_Z_3D);
#endif
  formats.export_values();
  m.def("get_native_memory_format", dipu::get_native_memory_format);
  m.def("native_memory_format_cast", dipu::native_memory_format_cast);
}

void exportGenerator(py::module& m) {
  m.def("_manual_seed", manual_seed);
  m.def("_seed", seed);
  m.def("_initial_seed", initial_seed);
  m.def("_get_rng_state", get_rng_state);
  m.def("_set_rng_state", set_rng_state);
  m.def("_is_in_bad_fork", is_in_bad_fork);
  m.def("_create_dipu_generator", [](int idx) -> at::Generator {
    auto index = static_cast<at::DeviceIndex>(idx);
    return createDIPUGenerator(index);
  });
}

void exportAutocast(py::module& m) {
  m.def("get_autocast_dipu_dtype", at::autocast::get_autocast_xpu_dtype);
  m.def("is_autocast_dipu_enabled", at::autocast::is_xpu_enabled);
  m.def("set_autocast_dipu_enabled", at::autocast::set_xpu_enabled);
  m.def("set_autocast_dipu_dtype", at::autocast::set_autocast_xpu_dtype);
}

void exportUtils(py::module& m) {
  m.def("get_dipu_torch_version", []() -> int { return DIPU_TORCH_VERSION; });
}

void exportMetrics(py::module& m) {
  using group = metrics::ExportedGroup;

  m.def("metrics", []() -> std::vector<group> {
    return group::from_collector(metrics::default_collector());
  });
  m.def("is_metrics_enabled", []() -> bool { return metrics::enable(); });
  m.def("enable_metrics", [](bool value) -> void { metrics::enable(value); });

  py::class_<group>(m, "MetricsGroup")
      .def(py::init<>())
      .def_readwrite("name", &group::name)
      .def_readwrite("type", &group::type)
      .def_readwrite("info", &group::info)
      .def_readwrite("values", &group::values)
      .def("asdict", [](group const& x) -> py::dict {
        // NOLINTNEXTLINE(google-build-using-namespace)
        using namespace py::literals;
        return py::dict("name"_a = x.name, "type"_a = x.type, "info"_a = x.info,
                        "values"_a = x.values);
      });
}

}  // namespace

extern void patchTorchCsrcDevice(py::module& m);
extern void patchTorchTensor(py::module& m);

DIPU_API void exportDIPURuntime(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  patchTorchCsrcDevice(m);
  patchTorchTensor(m);
  exportDevices(m);
  exportStream(m);
  exportEvent(m);
  exportCommunicator(m);
  exportMemCaching(m);
  patchStorage(m);
  patchTensor(m);
  exportNativeMemoryFormat(m);
  exportGenerator(m);
  exportAutocast(m);
  exportUtils(m);
  exportMetrics(m);
}
}  // namespace dipu
