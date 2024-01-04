#include "profiler_kineto.h"

#include <algorithm>
#include <libkineto.h>

#include <c10/macros/Export.h>
#include <c10/util/C++17.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/overloaded.h>
#include <c10/util/variant.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <torch/csrc/profiler/events.h>
#include <torch/csrc/profiler/util.h>

#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include "csrc_dipu/utils/Log.h"

#include "collection.h"
#include "profiler.h"

namespace dipu {
namespace profile {

namespace {
inline int64_t getTimeUs() {
  auto constexpr scale = int64_t{1000};
  return torch::profiler::impl::getTime(true) / scale;
}

const std::set<libkineto::ActivityType> kCpuTypes{
    libkineto::ActivityType::CPU_OP,
    libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::PYTHON_FUNCTION,
};

using torch::autograd::profiler::experimental_event_t;
using torch::autograd::profiler::KinetoEvent;
using torch::autograd::profiler::post_process_t;
using torch::autograd::profiler::ProfilerResult;
using torch::profiler::impl::ActiveProfilerType;
using torch::profiler::impl::dtypesToStr;
using torch::profiler::impl::EventType;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::op_input_t;
using torch::profiler::impl::ProfilerState;
using torch::profiler::impl::ProfilerStateBase;
using torch::profiler::impl::PyExtraFieldsBase;
using torch::profiler::impl::Result;
using torch::profiler::impl::shapesToStr;
using torch::profiler::impl::stacksToStr;
using torch::profiler::impl::TensorMetadata;

auto shapesAndDtypes(const std::vector<op_input_t>& inputs) {
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::string> dtypes;
  for (const auto& i : inputs) {
    c10::visit(c10::overloaded(
                   [&](const TensorMetadata& t) {
                     shapes.emplace_back(t.sizes_);
                     dtypes.emplace_back(scalarTypeToTypeMeta(t.dtype_).name());
                   },
                   [&](const std::vector<TensorMetadata>&) {
                     shapes.emplace_back();
                     dtypes.emplace_back("TensorList");
                   },
                   [&](const c10::IValue&) {
                     shapes.emplace_back();
                     dtypes.emplace_back("Scalar");
                   },
                   [&](const auto&) {
                     shapes.emplace_back();
                     dtypes.emplace_back();
                   }),
               i);
  }
  return std::make_pair(shapes, dtypes);
}

struct MetadataBase {
  explicit MetadataBase(const std::shared_ptr<Result>& result)
      : kineto_activity_{result->kineto_activity_} {
    if (c10::holds_alternative<ExtraFields<EventType::Kineto>>(
            result->extra_fields_)) {
      // In order to add metadata we have to downcast from
      // `libkineto::ITraceActivity` to `libkineto::GenericTraceActivity`. We
      // know that all activities provided by PyTorch are of the correct type,
      // however Kineto profilers can (and do) add events that inherit directly
      // from ITraceActivity. As a result, any Result which was constructed from
      // an event that Kineto provided is unsafe to cast.
      if (!(SOFT_ASSERT(!hasKinetoActivity()))) {
        result->kineto_activity_ = nullptr;
      }
      kineto_activity_ = result->kineto_activity_;
    }
  }

  void addMetadata(const std::string& key, const std::string& value) {
    if (kineto_activity_ && !value.empty() && value != "\"\"") {
      torch::profiler::impl::kineto::addMetadata(kineto_activity_, key, value);
    }
  }

  bool hasKinetoActivity() const { return kineto_activity_ != nullptr; }

 private:
  const torch::profiler::impl::kineto::activity_t* kineto_activity_{nullptr};
};

struct AddTensorboardFields : public MetadataBase {
  AddTensorboardFields(const std::shared_ptr<Result>& result,
                       KinetoEvent& kineto_event)
      : MetadataBase(result) {
    result->visit(*this);
    const auto module_hierarchy = kineto_event.moduleHierarchy();
    addMetadata("Module Hierarchy", stacksToStr(module_hierarchy.vec(), "."));
    addMetadata("Call stack", stacksToStr(kineto_event.stack().vec(), ";"));

    result->visit_if_base<PyExtraFieldsBase>([&, this](const auto& i) -> void {
      this->addMetadata("Python id", std::to_string(i.id_));

      c10::optional<std::string> parent_id;
      std::shared_ptr<Result> parent = result->parent_.lock();
      while (parent && !parent_id.has_value()) {
        parent->visit_if_base<PyExtraFieldsBase>(
            [&](const auto& j) { parent_id = std::to_string(j.id_); });
        parent = parent->parent_.lock();
      }
      this->addMetadata("Python parent id", parent_id.value_or("null"));
    });
  }

  void operator()(const ExtraFields<EventType::PyCall>& py_call) {
    if (py_call.module_.has_value()) {
      addMetadata("Python module id", std::to_string(py_call.module_->id_));
    }
  }

  template <typename T>
  void operator()(const T& /*unused*/) {}
};

struct AddGenericMetadata : public MetadataBase {
  AddGenericMetadata(std::shared_ptr<Result>& result,
                     const torch::profiler::impl::ProfilerConfig* config)
      : MetadataBase(result), config_(config) {
    result->visit(*this);
    if (config->experimental_config.verbose) {
      result->visit_if_base<PyExtraFieldsBase>(
          [&, this](const auto& i) -> void {
            this->addMetadata("Python thread", std::to_string(i.python_tid_));
          });
    }
  }

  void operator()(ExtraFields<EventType::TorchOp>& op_event) {
    const auto shapes_and_dtypes = shapesAndDtypes(op_event.inputs_);
    if (!shapes_and_dtypes.first.empty()) {
      addMetadata("Input Dims", shapesToStr(shapes_and_dtypes.first));
    }

    if (!shapes_and_dtypes.second.empty()) {
      addMetadata("Input type", dtypesToStr(shapes_and_dtypes.second));
    }

    if (config_ && !config_->experimental_config.performance_events.empty()) {
      auto& event_names = config_->experimental_config.performance_events;
      for (auto i = 0; i < op_event.perf_event_counters_->size(); ++i) {
        addMetadata(event_names[i],
                    std::to_string((*op_event.perf_event_counters_)[i]));
      }
    }

    // add information about an associated forward op, if a sequence number
    // is available (e.g. during training)
    if (op_event.sequence_number_ >= 0) {
      addMetadata("Fwd thread id", std::to_string(op_event.forward_tid_));
      addMetadata("Sequence number", std::to_string(op_event.sequence_number_));
    }
  }

  void operator()(ExtraFields<EventType::Backend>& backend_event) {
    if (!backend_event.backend_.empty()) {
      addMetadata("Backend", "\"" + backend_event.backend_ + "\"");
    }
  }

  void operator()(const ExtraFields<EventType::Allocation>& alloc) {
    addMetadata("Device Type",
                std::to_string(static_cast<int8_t>(alloc.device_type_)));
    addMetadata("Device Id", std::to_string(alloc.device_index_));
    addMetadata("Addr", std::to_string(reinterpret_cast<intptr_t>(alloc.ptr_)));
    addMetadata("Bytes", std::to_string(alloc.alloc_size_));
    addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
    addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
  }

  void operator()(const ExtraFields<EventType::OutOfMemory>& alloc) {
    addMetadata("Device Type",
                std::to_string(static_cast<int8_t>(alloc.device_type_)));
    addMetadata("Device Id", std::to_string(alloc.device_index_));
    addMetadata("Bytes", std::to_string(alloc.alloc_size_));
    addMetadata("Total Allocated", std::to_string(alloc.total_allocated_));
    addMetadata("Total Reserved", std::to_string(alloc.total_reserved_));
  }

  template <typename T>
  void operator()(const T& /*unused*/) {}

 private:
  /* To get names of the performance events */
  const torch::profiler::impl::ProfilerConfig* config_;
};
// Assumption: Total threads number will not exceed 2^16-1, and total ops will
// not exceed 2^48 -1.
inline uint64_t getForwardThreadKey(uint64_t tid, uint64_t seqNr) {
  auto constexpr shift = 48;
  auto constexpr mask = (uint64_t{1} << shift) - 1;
  return (tid << shift) | (seqNr & mask);
}

struct DIPUKinetoThreadLocalState : public ProfilerStateBase {
  explicit DIPUKinetoThreadLocalState(
      const torch::profiler::impl::ProfilerConfig& config,
      std::set<torch::profiler::impl::ActivityType> activities)
      : ProfilerStateBase(config),
        start_time_(getTimeUs()),
        record_queue_(config, std::move(activities)) {}
  ~DIPUKinetoThreadLocalState() override = default;

  static DIPUKinetoThreadLocalState* get(bool global) {
    auto* state = ProfilerStateBase::get(global);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(state == nullptr ||
                                     state->profilerType() ==
                                         ActiveProfilerType::KINETO);
    return static_cast<DIPUKinetoThreadLocalState*>(state);
  }

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::KINETO;
  }

  void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                         size_t total_reserved, c10::Device device) override {
    if (config_.profile_memory && !config_.disabled()) {
      record_queue_.getSubqueue()->emplace_allocation_event(
          torch::profiler::impl::getApproximateTime(), ptr, alloc_size,
          total_allocated, total_reserved, device.type(), device.index());
    }
  }

  void reportOutOfMemory(int64_t alloc_size, size_t total_allocated,
                         size_t total_reserved, c10::Device device) override {
    if (config_.profile_memory && !config_.disabled()) {
      record_queue_.getSubqueue()->emplace_ooms_event(
          torch::profiler::impl::getApproximateTime(), alloc_size,
          total_allocated, total_reserved, device.type(), device.index());
    }
  }

  const post_process_t& getEventPostProcessingCallback() const {
    return event_post_process_cb_;
  }

  void setEventPostProcessingCallback(post_process_t&& cb) {
    event_post_process_cb_ = std::move(cb);
  }

  std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>
  finalizeTrace() {
    auto end_time = getTimeUs();
    record_queue_.stop();

    std::lock_guard<std::mutex> guard(state_mutex_);
    auto converter = clock_converter_.makeConverter();
    auto records_and_trace =
        record_queue_.getRecords(std::move(converter), start_time_, end_time);

    materializeOpEvents(records_and_trace.first);

    // `kineto_events_` does not include Python events. Instead it exposes them
    // via the `stacks` property.
    kineto_events_.erase(
        std::remove_if(kineto_events_.begin(), kineto_events_.end(),
                       [](const auto& i) { return i.isPythonFunction(); }),
        kineto_events_.end());

    return std::move(records_and_trace.second);
  }

  template <typename T>
  void invokeCallback(T& t) {
    if (event_post_process_cb_) {
      event_post_process_cb_(t.debug_handle_, t.jit_stack_, t.jit_modules_);
    }
  }

  void materializeOpEvents(std::vector<std::shared_ptr<Result>>& events) {
    for (auto& e : events) {
      if (e->parent_.expired()) {
        event_tree_.push_back(e);
      }

      if (e->finished_) {
        e->visit(c10::overloaded(
            [this](ExtraFields<EventType::TorchOp>& i) { invokeCallback(i); },
            [this](ExtraFields<EventType::Backend>& i) { invokeCallback(i); },
            [](auto&) {}));

        kineto_events_.emplace_back(e, config_.experimental_config.verbose);
        AddTensorboardFields add_tb(e, kineto_events_.back());
        AddGenericMetadata add_generic(e, &config_);

        // It is not safe to use the activity after post processing.
        e->kineto_activity_ = nullptr;
      }
    }
  }

  static void generateForwardBackwardLink(
      const KinetoEvent& kineto_event, uint64_t& fwd_bwd_link_id,
      libkineto::GenericTraceActivity& activity,
      std::unordered_map<uint64_t, libkineto::GenericTraceActivity*>&
          tidSeq2activity) {
    if (kineto_event.fwdThreadId() > 0) {
      // act is backward op.
      uint64_t key = getForwardThreadKey(kineto_event.fwdThreadId(),
                                         kineto_event.sequenceNr());
      auto iter = tidSeq2activity.find(key);
      if (iter != tidSeq2activity.end()) {
        libkineto::GenericTraceActivity* fwd = iter->second;
        fwd->flow.start = true;
        activity.flow.id = fwd->flow.id = fwd_bwd_link_id;
        activity.flow.type = fwd->flow.type = libkineto::kLinkFwdBwd;
        ++fwd_bwd_link_id;
      }
    } else if (kineto_event.startThreadId() != 0) {
      // act is forward op.
      uint64_t key = getForwardThreadKey(kineto_event.startThreadId(),
                                         kineto_event.sequenceNr());
      // Assumption: Among all ops with same sequence number,
      // the one with biggest start time is most likely launching backward op.
      auto iter = tidSeq2activity.find(key);
      if (iter == tidSeq2activity.end()) {
        tidSeq2activity[key] = &activity;
      } else {
        // Now the sequence number is only incremented on creating a "Node"
        // object for backward pass, by calling
        // "at::sequence_number::get_and_increment()". Among all ops with same
        // sequence number, the one with biggest startTime is the one launching
        // backward op.
        if (activity.startTime >= iter->second->startTime) {
          tidSeq2activity[key] = &activity;
        }
      }
    }
  }

  uint64_t start_time_;
  torch::profiler::impl::ApproximateClockToUnixTimeConverter clock_converter_;
  DIPURecordQueue record_queue_;
  std::vector<KinetoEvent> kineto_events_;
  std::vector<experimental_event_t> event_tree_;
  // Optional, if event post-processing is enabled.
  post_process_t event_post_process_cb_;
};

template <bool use_global_state_ptr = false>
std::unique_ptr<at::ObserverContext> onFunctionEnter(
    const at::RecordFunction& fn) {
  auto state_ptr = DIPUKinetoThreadLocalState::get(use_global_state_ptr);
  if (!state_ptr) {
    return nullptr;
  }
  return state_ptr->record_queue_.getSubqueue()->begin_op(fn);
}

// @lint-ignore CLANGTIDY clang-diagnostic-unused-parameter
template <bool use_global_state_ptr = false>
void onFunctionExit(const at::RecordFunction& fn,
                    at::ObserverContext* ctx_ptr) {
  auto state_ptr = DIPUKinetoThreadLocalState::get(use_global_state_ptr);
  if (!state_ptr) {
    return;
  }
  const auto& config = state_ptr->config();
  auto* kineto_ctx_ptr =
      static_cast<torch::profiler::impl::KinetoObserverContext*>(ctx_ptr);
  TORCH_INTERNAL_ASSERT(kineto_ctx_ptr != nullptr);
  kineto_ctx_ptr->event_->end_time_ =
      torch::profiler::impl::getApproximateTime();
  if (!config.experimental_config.performance_events.empty()) {
    state_ptr->record_queue_.getSubqueue()->disable_perf_profiler(
        *kineto_ctx_ptr->event_->counters_);
  }
  kineto_ctx_ptr->event_->basic_fields_.end_tid_ =
      at::RecordFunction::currentThreadId();

  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    libkineto::api().activityProfiler().popUserCorrelationId();
  } else {
    libkineto::api().activityProfiler().popCorrelationId();
  }
}

template <bool use_global_callback = false>
void pushProfilingCallbacks(const std::unordered_set<at::RecordScope>& scopes) {
  auto registration_state_ptr =
      DIPUKinetoThreadLocalState::get(use_global_callback);
  TORCH_INTERNAL_ASSERT(registration_state_ptr, "Expected profiler state set");
  auto recordFunctionCallback =
      at::RecordFunctionCallback(onFunctionEnter<use_global_callback>,
                                 onFunctionExit<use_global_callback>)
          .needsInputs(registration_state_ptr->config().report_input_shapes)
          .scopes(scopes);

  auto handle = c10::guts::if_constexpr<use_global_callback>(
      [&] { return at::addGlobalCallback(recordFunctionCallback); },
      [&] { return at::addThreadLocalCallback(recordFunctionCallback); });
  registration_state_ptr->setCallbackHandle(handle);
}

}  // namespace

static void prepareTrace(
    const bool cpuOnly,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const torch::profiler::impl::ExperimentalConfig& config) {
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) {
    libkineto::api().initProfilerIfRegistered();
  }

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(torch::profiler::impl::ActivityType::CPU) != 0U) {
    k_activities.insert(kCpuTypes.begin(), kCpuTypes.end());
  }
  if (activities.count(torch::profiler::impl::ActivityType::CUDA) != 0U) {
    k_activities.insert(libkineto::ActivityType::CONCURRENT_KERNEL);
  }
  libkineto::api().activityProfiler().prepareTrace(k_activities);
}

void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities) {
  TORCH_CHECK(config.state == ProfilerState::KINETO ||
                  config.state == ProfilerState::KINETO_GPU_FALLBACK,
              "Supported only in Kineto profiler");

  bool cpuOnly = (devproxy::getDeviceCount() <= 0);
  prepareTrace(cpuOnly, activities, config.experimental_config);

  if (!config.experimental_config.performance_events.empty()) {
    /* For now only CPU activity is supported */
    TORCH_CHECK(activities.count(torch::profiler::impl::ActivityType::CPU),
                "Cannot run cpu hardware profiler without CPU activities, "
                "please only use CPU activity type");
    /*
     * Sending a warning and passing the non-standard event to the backend
     * Backend can abort if the event is not supported.
     * TODO Should we gracefully drop the invalid event if we have atleast one
     * valid?
     */
    auto is_standard_event = [](const std::string& event) -> bool {
      auto equal = [&event](const char* str) { return event == str; };
      return std::any_of(torch::profiler::ProfilerPerfEvents.begin(),
                         torch::profiler::ProfilerPerfEvents.end(), equal);
    };

    for (const auto& e : config.experimental_config.performance_events) {
      if (!is_standard_event(e)) {
        TORCH_WARN("Forwarding a non-standard CPU performance event : ", e);
      }
    }
  }
}
void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes) {
  const auto has_cpu =
      activities.count(torch::profiler::impl::ActivityType::CPU);
  TORCH_CHECK(
      DIPUKinetoThreadLocalState::get(/*global=*/config.global()) == nullptr,
      "Profiler is already enabled",
      (config.global() ? "." : " on this thread."));

  TORCH_CHECK(config.state == ProfilerState::KINETO || config.global());
  TORCH_CHECK(!activities.empty(), "No activities specified.");
  TORCH_INTERNAL_ASSERT(has_cpu || !config.global(),
                        "Ondemand profiling must enable CPU tracing");

  DIPUKinetoThreadLocalState::push(
      std::make_shared<DIPUKinetoThreadLocalState>(config, activities));

  if (has_cpu) {
    config.global() ? pushProfilingCallbacks</*global=*/true>(scopes)
                    : pushProfilingCallbacks</*global=*/false>(scopes);
  }

  if (!config.global()) {
    libkineto::api().activityProfiler().startTrace();
  }

  const auto has_device =
      activities.count(torch::profiler::impl::ActivityType::CUDA);
  if (has_device) {
    setProfileOpen(true);
  }
}

std::unique_ptr<ProfilerResult> disableProfiler() {
  auto state_ptr = ProfilerStateBase::pop();
  const auto& config = state_ptr->config();
  TORCH_CHECK(state_ptr && (config.state == ProfilerState::KINETO),
              "Can't disable Kineto profiler when it's not running");

  state_ptr->removeCallback();

  // Traces are converged via libkineto automatically for ondemand flow
  if (state_ptr->config().global()) {
    (void)std::static_pointer_cast<DIPUKinetoThreadLocalState>(state_ptr)
        ->finalizeTrace();
    return std::make_unique<ProfilerResult>();
  }

  std::unique_ptr<ProfilerResult> result;
  if (config.state == ProfilerState::KINETO) {
    auto kineto_state_ptr =
        std::static_pointer_cast<DIPUKinetoThreadLocalState>(state_ptr);
    auto trace = kineto_state_ptr->finalizeTrace();
    result = std::make_unique<ProfilerResult>(
        kineto_state_ptr->start_time_,
        std::move(kineto_state_ptr->kineto_events_), std::move(trace),
        std::move(kineto_state_ptr->event_tree_));
  }

  return result;
}

void addMetadataJson(const std::string& key, const std::string& value) {
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().addMetadata(key, value);
  } else {
    DIPU_LOG << "Profiler is not initialized: skipping profiling metadata"
             << std::endl;
  }
}

void profilerStep() {
  if (libkineto::api().isProfilerInitialized()) {
    libkineto::api().activityProfiler().step();
  } else {
    DIPU_LOG << "Profiler is not initialized: skipping step() invocation"
             << std::endl;
  }
}

}  // namespace profile
}  // namespace dipu
