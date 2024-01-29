#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/strong_type.h>
#include <c10/util/variant.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/data_flow.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/orchestration/python_tracer.h>
#include <torch/csrc/profiler/perf.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>
#include <torch/csrc/utils/python_stub.h>

namespace dipu {
namespace profile {

// DIPUInputOutputEncoder
// Stores each op_events' shapes and dtypes into a contiguous AppendOnlyList
// so that we no longer create vectors for shapes and dtypes on every op.
// Those vectors can be created during post-processing.
class DIPUInputOutputEncoder final {
 public:
  void push(c10::ArrayRef<const c10::IValue> values);

  auto getInputShapeGenerator();
  auto getConcreteInputGenerator();
  static bool isSupportedScalarList(const c10::IValue& list_candidate);

  void clear();

  enum class Tag {
    Tensor = 0,
    UndefinedTensor,
    TensorListBegin,  // TODO(caikun-pjlab): generalize to other lists.
    ScalarList,
    Scalar,
    Other,
    TERMINATOR
  };

  enum class IOType { Shapes, ConcreteInputs, None };

 private:
  void push(const at::Tensor& t);

  // Implementation detail for getInputShapeGenerator and
  // getConcreteInputGenerator
  auto getIValueGenerator(const IOType& io_type);

  torch::profiler::impl::AppendOnlyList<
      Tag, torch::profiler::impl::IO_ENCODER_DEFAULT_BLOCK_SIZE>
      tags_;
  torch::profiler::impl::AppendOnlyList<
      torch::profiler::impl::RawTensorMetadata,
      torch::profiler::impl::IO_ENCODER_DEFAULT_BLOCK_SIZE>
      tensor_metadata_;
  torch::profiler::impl::AppendOnlyList<
      int64_t, torch::profiler::impl::IO_ENCODER_DEFAULT_BLOCK_SIZE>
      tensor_sizes_strides_;
  torch::profiler::impl::AppendOnlyList<
      c10::IValue, torch::profiler::impl::IO_ENCODER_DEFAULT_BLOCK_SIZE>
      ivalues_;
};

class DIPUThreadLocalSubqueue {
 public:
  DIPUThreadLocalSubqueue(uint64_t tid,
                          const torch::profiler::impl::ProfilerConfig& config);

  std::unique_ptr<torch::profiler::impl::KinetoObserverContext> begin_op(
      const at::RecordFunction& fn);

  template <class... Args>
  void emplace_backend_event(Args&&... args) {
    backend_events_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_vulkan_event(Args&&... args) {
    vulkan_events_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_allocation_event(Args&&... args) {
    allocations_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_ooms_event(Args&&... args) {
    ooms_.emplace_back(std::forward<Args>(args)...);
  }

  template <class... Args>
  void emplace_py_call(Args&&... args) {
    py_calls_.emplace_back(std::forward<Args>(args)...);
  }

  uint64_t tid() const { return tid_; }

  const torch::profiler::impl::kineto::DeviceAndResource& kineto_info() const {
    return kineto_info_;
  }

  inline void disable_perf_profiler(
      torch::profiler::perf_counters_t& counters) const {
    perf_profiler_->Disable(counters);
  }

 private:
  uint64_t tid_;
  torch::profiler::impl::ProfilerConfig config_;
  torch::profiler::impl::kineto::DeviceAndResource kineto_info_;
  std::unique_ptr<torch::profiler::impl::perf_profiler_t> perf_profiler_;

  friend class DIPURecordQueue;
  // See `containers.h` for block size benchmarks.
  static constexpr size_t BlockSize = 512;

  struct TorchOpStorage {
    // NB: This is a destructive operation.
    void materialize(
        std::vector<std::shared_ptr<torch::profiler::impl::Result>>& out,
        const std::function<time_t(torch::profiler::impl::approx_time_t)>&
            time_converter,
        uint64_t tid,
        const torch::profiler::impl::kineto::DeviceAndResource& kineto_info);

    template <typename T, size_t ChunkSize>
    class EventBlock : public std::array<T, ChunkSize> {
     public:
      EventBlock();
      uint64_t correlation_id(const T* ptr) const;

     private:
      uint64_t id_start_;
    };

    using event_t = torch::profiler::impl::KinetoObserverContext::Event;
    class OpList
        : public torch::profiler::impl::AppendOnlyList<event_t, BlockSize,
                                                       EventBlock> {
     public:
      template <class... Args>
      std::pair<event_t*, uint64_t> emplace_back(Args&&... args);
      static uint64_t correlationID(const OpList::Iterator& e);
    } op_events_;

    // report_input_shapes
    DIPUInputOutputEncoder inputs_outputs_;

    // with_stack (JIT)
    torch::profiler::impl::AppendOnlyList<torch::profiler::impl::jit_stack_t,
                                          BlockSize>
        jit_stack_;

    // with_modules
    torch::profiler::impl::AppendOnlyList<torch::profiler::impl::jit_modules_t,
                                          BlockSize>
        jit_modules_;

    // with_flops
    torch::profiler::impl::AppendOnlyList<torch::profiler::impl::extra_args_t,
                                          BlockSize>
        extra_args_;

    // ProfilerState::KINETO_GPU_FALLBACK
    torch::profiler::impl::AppendOnlyList<torch::profiler::impl::FallbackPair,
                                          BlockSize>
        gpu_fallback_;
  } torch_ops_;

  // reportBackendEventToActiveKinetoProfiler
  torch::profiler::impl::AppendOnlyList<
      torch::profiler::impl::ExtraFields<
          torch::profiler::impl::EventType::Backend>,
      BlockSize>
      backend_events_;

  // _reportVulkanEventToProfiler
  torch::profiler::impl::AppendOnlyList<
      torch::profiler::impl::ExtraFields<
          torch::profiler::impl::EventType::Vulkan>::raw_event_t,
      BlockSize>
      vulkan_events_;

  // reportMemoryUsage
  torch::profiler::impl::AppendOnlyList<torch::profiler::impl::RawAllocation,
                                        BlockSize>
      allocations_;

  // reportOOMs
  torch::profiler::impl::AppendOnlyList<
      torch::profiler::impl::ExtraFields<
          torch::profiler::impl::EventType::OutOfMemory>,
      BlockSize>
      ooms_;

  // with_stack (Python)
  torch::profiler::impl::AppendOnlyList<
      std::pair<torch::profiler::impl::python_tracer::TraceKey,
                torch::profiler::impl::approx_time_t>,
      BlockSize>
      py_calls_;
};

class DIPURecordQueue {
 public:
  DIPURecordQueue(const torch::profiler::impl::ProfilerConfig& config,
                  std::set<torch::profiler::impl::ActivityType> activities);

  bool tracePython() const;
  DIPUThreadLocalSubqueue* getSubqueue();
  void stop();

  // NB: This is a destructive operation.
  std::pair<
      std::vector<std::shared_ptr<torch::profiler::impl::Result>>,
      std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>>
  getRecords(std::function<time_t(torch::profiler::impl::approx_time_t)>
                 time_converter,
             uint64_t start_time_us, uint64_t end_time_us);

 private:
  uint32_t id_;
  torch::profiler::impl::ProfilerConfig config_;
  std::set<torch::profiler::impl::ActivityType> activities_;
  ska::flat_hash_map<uint64_t, std::unique_ptr<DIPUThreadLocalSubqueue>>
      sub_queues_;
  std::mutex sub_queue_mutex_;
  std::unique_ptr<torch::profiler::impl::python_tracer::PythonTracerBase>
      python_tracer_;
};

}  // namespace profile
}  // namespace dipu
