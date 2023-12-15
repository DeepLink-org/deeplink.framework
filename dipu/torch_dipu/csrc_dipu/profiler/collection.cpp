#include "collection.h"

#include <algorithm>
#include <fmt/format.h>
#include <functional>
#include <libkineto.h>
#include <limits>
#include <memory>
#include <queue>
#include <type_traits>
#include <utility>

#include <ATen/Context.h>
#include <ATen/record_function.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/hash.h>
#include <c10/util/overloaded.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/profiler/data_flow.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/orchestration/vulkan.h>

#include "profiler_python.h"

namespace dipu {
namespace profile {

constexpr bool kKinetoAvailable{true};

using torch::profiler::perf_counters_t;
using torch::profiler::impl::ActivityType;
using torch::profiler::impl::approx_time_t;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::KinetoObserverContext;
using torch::profiler::impl::op_input_t;
using torch::profiler::impl::ProfilerConfig;
using torch::profiler::impl::Result;
using torch::profiler::impl::TensorMetadata;
using torch::profiler::impl::kineto::ActivityTraceWrapper;
using torch::profiler::impl::kineto::DeviceAndResource;
using torch::profiler::impl::kineto::interface_trace_t;
using torch::profiler::impl::kineto::kineto_ids;
using torch::profiler::impl::python_tracer::CompressedEvent;

using result_ptr_t = std::shared_ptr<torch::profiler::impl::Result>;
using trace_ptr_t =
    std::unique_ptr<torch::profiler::impl::kineto::ActivityTraceWrapper>;

void DIPUInputOutputEncoder::push(c10::ArrayRef<const c10::IValue> values) {
  for (const auto& value : values) {
    if (value.isTensor()) {
      push(value.toTensor());
    } else if (value.isScalar()) {
      tags_.emplace_back(Tag::Scalar);
      // Scalars are small enough that they are stored in ivalues without an
      // extra memory alloc
      // TODO(caikun-pjlab): further optimize this by maybe giving Profiler
      // access to the guts of IValue.
      ivalues_.emplace_back(value);
    } else if (value.isTensorList()) {
      tags_.emplace_back(Tag::TensorListBegin);
      for (const auto& t : value.toTensorList()) {
        push(t);
      }
      tags_.emplace_back(Tag::TERMINATOR);
    } else {
      tags_.emplace_back(Tag::Other);
    }
  }
  tags_.emplace_back(Tag::TERMINATOR);
}

void DIPUInputOutputEncoder::push(const at::Tensor& t) {
  if (t.defined() && !t.is_nested()) {  // TODO(caikun-pjlab) fix nested sizes
    tags_.emplace_back(Tag::Tensor);
    tensor_metadata_.emplace_back(t);
    tensor_sizes_strides_.copy(t.sizes());
    if (t.layout() == at::kStrided) {
      // Only Strided layout tensors have strides
      tensor_sizes_strides_.copy(t.strides());
    }
  } else {
    tags_.emplace_back(Tag::UndefinedTensor);
  }
}

// This is a custom-iterator-like getter to obtain input shapes and dtypes.
auto DIPUInputOutputEncoder::getNextShapesAndDtypes() {
  return [this, tag_it = tags_.begin(),
          tensor_metadata_it = tensor_metadata_.begin(),
          tensor_size_strides_it = tensor_sizes_strides_.begin(),
          ivals_it = ivalues_.begin()]() mutable {
    auto decode_tensor = [&]() -> TensorMetadata {
      const auto& raw_metadata = *tensor_metadata_it++;
      std::vector<int64_t> sizes;
      std::vector<int64_t> strides;
      for (C10_UNUSED const auto _ : c10::irange(raw_metadata.dim_)) {
        sizes.push_back(*tensor_size_strides_it++);
      }
      if (raw_metadata.layout_ == at::kStrided) {
        for (C10_UNUSED const auto _ : c10::irange(raw_metadata.dim_)) {
          strides.push_back(*tensor_size_strides_it++);
        }
      }
      return {raw_metadata, sizes, strides};
    };

    std::vector<op_input_t> out;
    bool terminate = false;
    while (!terminate && tag_it != tags_.end()) {
      switch (*tag_it) {
        case Tag::Tensor:
          out.emplace_back(decode_tensor());
          break;

        case Tag::TensorListBegin: {
          std::vector<TensorMetadata> arg;
          while (*(++tag_it) != Tag::TERMINATOR) {
            TORCH_INTERNAL_ASSERT(*tag_it == Tag::Tensor, (int)(*tag_it));
            arg.emplace_back(decode_tensor());
          }
          out.emplace_back(std::move(arg));
        } break;

        case Tag::Scalar:
          out.emplace_back(*ivals_it++);
          break;

        case Tag::UndefinedTensor:
        case Tag::Other:
          out.emplace_back(c10::nullopt);
          break;

        case Tag::TERMINATOR:
          // This marks the end of this op.
          terminate = true;
          break;

        default:
          break;
      }
      ++tag_it;
    }
    return out;
  };
}

void DIPUInputOutputEncoder::clear() {
  tags_.clear();
  tensor_metadata_.clear();
  tensor_sizes_strides_.clear();
  ivalues_.clear();
}

// ---------------------------------------------------
// |  Correlation ID tracking (OpList & EventBlock)  |
// ---------------------------------------------------
template <typename T, size_t ChunkSize>
DIPUThreadLocalSubqueue::TorchOpStorage::EventBlock<T,
                                                    ChunkSize>::EventBlock() {
  static std::atomic<uint64_t> counter_{0};
  id_start_ = 1 + ChunkSize * counter_++;
}

template <class... Args>
std::pair<KinetoObserverContext::Event*, uint64_t>
DIPUThreadLocalSubqueue::TorchOpStorage::OpList::emplace_back(Args&&... args) {
  maybe_grow();
  *next_ = {std::forward<Args>(args)...};
  auto corr_id = buffer_last_->correlation_id(next_);
  return {next_++, corr_id};
}

uint64_t DIPUThreadLocalSubqueue::TorchOpStorage::OpList::correlationID(
    const OpList::Iterator& e) {
  return e.address().first->correlation_id(&*e);
}

template <typename T, size_t ChunkSize>
uint64_t DIPUThreadLocalSubqueue::TorchOpStorage::EventBlock<
    T, ChunkSize>::correlation_id(const T* ptr) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ptr >= this->data() &&
                                   ptr < this->data() + ChunkSize);
  return id_start_ + (ptr - this->data());
}

// ---------------------------------
// |  Collection (Observer logic)  |
// ---------------------------------
std::unique_ptr<KinetoObserverContext> DIPUThreadLocalSubqueue::begin_op(
    const at::RecordFunction& fn) {
  KinetoObserverContext::Event* event = nullptr;
  uint64_t corr_id = 0;
  std::tie(event, corr_id) = torch_ops_.op_events_.emplace_back(
      fn.seqNr(), fn.forwardThreadId(), fn.scope(), fn.isAsync(),
      fn.debugHandle(), fn.name());
  if (config_.report_input_shapes) {
    torch_ops_.inputs_outputs_.push(fn.inputs());
  }

  if (fn.scope() == at::RecordScope::USER_SCOPE) {
    libkineto::api().activityProfiler().pushUserCorrelationId(corr_id);
  } else {
    libkineto::api().activityProfiler().pushCorrelationId(corr_id);
  }

  // backward nodes source range corresponds to the forward node
  // TODO(caikun-pjlab): consider using C++ stack trace
  if (config_.with_stack && fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    auto cs =
        torch::profiler::impl::prepareCallstack(torch::jit::currentCallstack());
    torch_ops_.jit_stack_.emplace_back(callstackStr(cs));
  }
  if (config_.with_modules &&
      fn.scope() != at::RecordScope::BACKWARD_FUNCTION) {
    torch_ops_.jit_modules_.emplace_back(torch::jit::currentModuleHierarchy());
  }
  if (config_.with_flops) {
    torch_ops_.extra_args_.emplace_back(
        torch::profiler::impl::saveExtraArgs(fn));
  }

  auto out = std::make_unique<KinetoObserverContext>(event);
  event->start_time_ = torch::profiler::impl::getApproximateTime();
  event->allow_tf32_cublas_ = at::globalContext().allowTF32CuBLAS();
  if (!config_.experimental_config.performance_events.empty()) {
    const size_t n = config_.experimental_config.performance_events.size();
    event->counters_ = std::make_unique<perf_counters_t>(n, 0);
    perf_profiler_->Enable();
  }
  return out;
}

// ---------------
// |  Collation  |
// ---------------
namespace {
template <typename T>
struct StealOrDefault {
  explicit StealOrDefault(T& container)
      : container_{container}, it_{container.begin()} {}

  ~StealOrDefault() { container_.get().clear(); }

  typename T::Iterator::value_type operator()() {
    if (it_.exhausted()) {
      return typename T::Iterator::value_type();
    }
    auto result = std::move(*it_);
    ++it_;
    return result;
  }

  std::reference_wrapper<T> container_;
  typename T::Iterator it_;
};
}  // namespace

void DIPUThreadLocalSubqueue::TorchOpStorage::materialize(
    std::vector<std::shared_ptr<Result>>& out,
    const std::function<time_t(approx_time_t)>& time_converter, uint64_t tid,
    const DeviceAndResource& kineto_info) {
  // Plumb Autograd info to the top level annotation.
  auto it = op_events_.begin();
  for (C10_UNUSED const auto _ :
       c10::irange(static_cast<int64_t>(op_events_.size()) - 1)) {
    auto& first = it->basic_fields_;
    auto& second = (++it)->basic_fields_;
    if (first.scope_ == at::RecordScope::FUNCTION &&
        second.scope_ == at::RecordScope::BACKWARD_FUNCTION &&
        first.name_.rfind("autograd::engine::evaluate_function: ", 0) == 0) {
      first.sequence_number_ = second.sequence_number_;
      first.forward_tid_ = second.forward_tid_;
    }
  }

  // `AccumulateGrad` is an important marker for profile analysis; however the
  // annotation relies on `c10::demangle` which is platform dependent. In
  // particular, Windows will add a "struct " prefix.
  const std::string accumulate_grad = "torch::autograd::AccumulateGrad";
  const std::string windows_pattern = std::string("struct ") + accumulate_grad;
  for (auto& event : op_events_) {
    auto& name = event.basic_fields_.name_;
    auto position = name.find(windows_pattern);
    if (position != std::string::npos) {
      name.replace(position, windows_pattern.size(), accumulate_grad);
    }
  }

  auto input_getter = inputs_outputs_.getNextShapesAndDtypes();

  // TODO(caikun-pjlab): CTAD will take care of template args when we move to
  // C++17
  auto jit_stack = StealOrDefault<decltype(jit_stack_)>(jit_stack_);
  auto jit_module = StealOrDefault<decltype(jit_modules_)>(jit_modules_);
  auto extra_args = StealOrDefault<decltype(extra_args_)>(extra_args_);
  auto gpu_fallback = StealOrDefault<decltype(gpu_fallback_)>(gpu_fallback_);

  for (auto event = op_events_.begin(); event != op_events_.end(); ++event) {
    ExtraFields<torch::profiler::impl::EventType::TorchOp> e{
        std::move(event->basic_fields_),
        DIPUThreadLocalSubqueue::TorchOpStorage::OpList::correlationID(event),
        time_converter(event->end_time_),
        input_getter(),
        jit_stack(),
        jit_module(),
        extra_args(),
        gpu_fallback(),
        event->allow_tf32_cublas_,
        std::move(event->counters_)};

    out.emplace_back(Result::create(time_converter(event->start_time_), tid,
                                    kineto_info, std::move(e)));
  }

  op_events_.clear();
  inputs_outputs_.clear();
}

namespace {
// See `DIPURecordQueue::getSubqueue()` for an overview of this cache.
struct SubQueueThreadCache {
  uint32_t key_;
  DIPUThreadLocalSubqueue* ref_;
};

// The astute observer will note that this leaves a dangling reference; nothing
// in the teardown of `DIPURecordQueue` or `DIPUThreadLocalSubqueue` clears this
// value. (And the raw pointer in `SubQueueThreadCache` will not extend the
// lifetime of `*ref_`.) This is safe, however, because `getSubqueue` will check
// `sub_queue_cache_.key_` before attempting to access `ref_`, and if `key_`
// does not match the DIPURecordQueue's *unique* `id_` it will evict
// `sub_queue_cache_` and fall back to a different mechanism.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<uint32_t> queue_id_{0};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local SubQueueThreadCache sub_queue_cache_{0, nullptr};

std::string toString(
    const ExtraFields<torch::profiler::impl::EventType::PyCall>& e) {
  if (e.module_.has_value()) {
    return fmt::format("nn.Module: {}_{}", e.module_->cls_name_.str(),
                       e.module_->id_);
  }
  return fmt::format("{}({}): {}", e.callsite_.filename_.str(),
                     e.callsite_.line_no_, e.callsite_.funcname_.str());
}

auto scopeToType(at::RecordScope scope) {
  return scope == at::RecordScope::USER_SCOPE
             ? libkineto::ActivityType::USER_ANNOTATION
             : libkineto::ActivityType::CPU_OP;
}

int64_t torchOpEndNS(
    const ExtraFields<torch::profiler::impl::EventType::TorchOp>& e,
    const bool finished, const std::weak_ptr<Result>& parent) {
  if (finished && e.end_time_ns_ == std::numeric_limits<time_t>::min()) {
    auto p = parent.lock();
    if (p) {
      return p->endTimeNS();
    }
  }
  return e.end_time_ns_;
}

auto kinetoEventCorrelationID(
    const ExtraFields<torch::profiler::impl::EventType::Kineto>& e,
    const std::weak_ptr<Result>& parent) {
  if (e.correlation_id_) {
    return e.correlation_id_;
  }
  auto p = parent.lock();
  return p ? p->correlationID() : 0;
}
}  // namespace

DIPUThreadLocalSubqueue::DIPUThreadLocalSubqueue(uint64_t tid,
                                                 const ProfilerConfig& config)
    : tid_{tid}, config_{config}, kineto_info_{kineto_ids()} {
  libkineto::api().activityProfiler().recordThreadInfo();
  if (!config_.experimental_config.performance_events.empty()) {
    perf_profiler_ =
        std::make_unique<torch::profiler::impl::linux_perf::PerfProfiler>();
    perf_profiler_->Configure(config_.experimental_config.performance_events);
  }
}

DIPURecordQueue::DIPURecordQueue(const ProfilerConfig& config,
                                 std::set<ActivityType> activities)
    : id_(++queue_id_), config_{config}, activities_{std::move(activities)} {
  if (tracePython()) {
    python_tracer_ = makeTracer(this);
  }
}

bool DIPURecordQueue::tracePython() const {
  return config_.with_stack && (activities_.count(ActivityType::CPU) != 0U);
}

DIPUThreadLocalSubqueue* DIPURecordQueue::getSubqueue() {
  // In the most common case, a thread will want to write to the same sub-queue
  // that it wrote to last call. The only time that isn't true is if:
  //  A) The profiler context has ended and we are in a new one.
  //  B) Two profilers are active in different TLS contexts, and this thread
  //     is a worker helping with intra-op parallelism.
  // Since we expect this to be the OVERWHELMINGLY common case (>99%), we add a
  // special thread_local cache so that we can skip the overall `flat_hash_map`
  // (and corresponding lock).
  if (id_ == sub_queue_cache_.key_) {
    return sub_queue_cache_.ref_;
  }

  const auto tid = at::RecordFunction::currentThreadId();
  std::lock_guard<std::mutex> guard(sub_queue_mutex_);
  auto it = sub_queues_.find(tid);
  if (it == sub_queues_.end()) {
    it = sub_queues_
             .emplace(tid,
                      std::make_unique<DIPUThreadLocalSubqueue>(tid, config_))
             .first;
  }

  sub_queue_cache_ = SubQueueThreadCache{id_, it->second.get()};
  return it->second.get();
}

void DIPURecordQueue::stop() {
  if (python_tracer_) {
    python_tracer_->stop();
  }
}

namespace {
void mark_finished(std::shared_ptr<Result>& r) {
  TORCH_INTERNAL_ASSERT(!r->finished_, r->name());
  r->finished_ = true;
  TORCH_INTERNAL_ASSERT(r->endTimeNS() >= r->start_time_ns_, r->name());
}

constexpr const char* indexKey = "Ev Idx";

void passEventsToKineto(const std::vector<std::shared_ptr<Result>>& results,
                        uint64_t start_time_us, uint64_t end_time_us) {
  using torch::profiler::impl::kineto::addMetadata;
  using torch::profiler::impl::kineto::TraceWrapper;
  constexpr time_t ns_per_us = 1000;
  TraceWrapper cpu_trace(static_cast<time_t>(start_time_us),
                         "PyTorch Profiler");

  // Generate Kineto events for each event recorded by the PyTorch profiler.
  for (const auto i : c10::irange(results.size())) {
    const auto& e = results[i];
    const auto* activity = cpu_trace.addCPUActivity(
        e->name(), e->kinetoType(), e->kineto_info_, e->correlationID(),
        e->start_time_ns_ / ns_per_us, e->endTimeNS() / ns_per_us);

    TORCH_INTERNAL_ASSERT(activity || !kKinetoAvailable);
    if (activity) {
      addMetadata(activity, indexKey, std::to_string(i));
    }
  }

  // Kineto adds the events that it collected.
  cpu_trace.transferCpuTrace(static_cast<int64_t>(end_time_us));
}

// There are two mechanisms that we use to connect Profiler and Kineto events.
// The first is the correlation ID. The profiler pushes a unique integer at the
// start of an op and pops it at the end. Kineto then associates the events
// that it collects with that correlation ID and sets the linked activity of
// the events that it collected to point to the profiler op.
//
// However, this is not a sufficient description because it does not retain
// dependency information between kineto ops. Consider a call to `torch.add`.
// Three events will be collected:
//   `aten::add`          (TorchOp, collected by profiler)
//   `cudaLaunchKernel`   (CUDA runtime event, collected by Kineto)
//   `at::vectorized_...` (GPU kernel, collected by Kineto)
// If we only relied on correlation IDs we would set both Kineto events as
// children of the `at::add`, rather than the correct
//   `at::add -> cudaLaunchKernel -> at::vectorized_...`
//
// Kineto surfaces this information through a second concept called a "flow".
// In this example, the `cudaLaunchKernel` event is the start of a flow and the
// GPU kernel has the same flow id but is not a start event. Thus, when merging
// the Kineto events into the call tree we first add all events which are flow
// start nodes. We then merge the rest, trying to pair them with flow starts
// and falling back to correlation ID if necessary. For any nodes without
// linked events the caller is determined using the normal tree construction
// algorithm.
class TransferEvents {
  using itrace_t = libkineto::ITraceActivity;
  using activity_t = torch::profiler::impl::kineto::activity_t;

 public:
  TransferEvents(std::vector<std::shared_ptr<Result>>& results,
                 trace_ptr_t& trace)
      : results_{results} {
    auto* trace_activities_ptr = trace->get()->activities();
    TORCH_INTERNAL_ASSERT(trace_activities_ptr != nullptr);
    trace_activities_ = *trace_activities_ptr;
    reassociate();
    extractEventsFromTrace();
    setParents();
  }

 private:
  static int64_t extractIndex(const std::string& metadata_json) {
    static const auto prefix = fmt::format("\"{}\": ", indexKey);
    auto pos = metadata_json.find(prefix);
    return (pos == std::string::npos) ? unmatchedIndex : [&]() {
      auto end = metadata_json.find(',', pos);
      end = (end == std::string::npos) ? metadata_json.size() : end;
      return std::stoll(metadata_json.substr(pos + prefix.size(), end));
    }();
  }

  std::shared_ptr<Result> lookup(const itrace_t* key) {
    if (key == nullptr) {
      return nullptr;
    }

    // First check the map.
    auto it = kineto_events_.find(key);
    if (it != kineto_events_.end()) {
      return it->second;
    }

    // Then fallback to the encoded metadata.
    const auto index = extractIndex(key ? key->metadataJson() : "");
    if (index != unmatchedIndex) {
      auto out = results_.get().at(index);
      kineto_events_[key] = out;
      return out;
    }
    // And finally give up.
    return nullptr;
  }

  void reassociate() {
    // Match profiler events with the corresponding kineto events. Kineto may
    // have moved or copied the activities, so we have to recover the
    // relationship between `libkineto::ITraceActivity` and `Result`.
    for (const auto* activity : trace_activities_) {
      TORCH_INTERNAL_ASSERT(activity != nullptr);
      auto e = lookup(activity);
      if (e != nullptr) {
        TORCH_INTERNAL_ASSERT(e->kineto_activity_ == nullptr);
        e->kineto_activity_ = static_cast<const activity_t*>(activity);
      }
    }
    if (results_.get().size() != kineto_events_.size()) {
      TORCH_WARN(
          fmt::format("Failed to recover relationship between all "
                      "profiler and kineto events: "
                      "{} vs. {}  reassociated.",
                      results_.get().size(), kineto_events_.size()));
    }
  }

  static std::shared_ptr<Result> resultFromActivity(const itrace_t* activity) {
    TORCH_INTERNAL_ASSERT(activity != nullptr);
    constexpr size_t ns_per_us = 1000;

    // Kineto is inconsistent with types, so we have to cast to int32.
    torch::profiler::impl::kineto::DeviceAndResource device_and_resource{
        static_cast<int32_t>(activity->deviceId()),
        static_cast<int32_t>(activity->resourceId())};

    auto event = Result::create(
        activity->timestamp() * ns_per_us,
        noTID,  // Placeholder
        device_and_resource,
        ExtraFields<torch::profiler::impl::EventType::Kineto>{
            activity->name(),
            activity->duration(),
            static_cast<uint64_t>(activity->correlationId()),
            activity->type(),
            {/*id=*/static_cast<uint32_t>(activity->flowId()),
             /*type=*/static_cast<uint32_t>(activity->flowType()),
             /*start=*/static_cast<uint32_t>(activity->flowStart())}});

    // NB: It's tempting to set `event->kineto_activity_`; however we can only
    // guarantee that the events we passed to Kineto are of type
    // `GenericTraceActivity`. Others may derive from ITraceActivity and thus
    // are not safe to cast.
    return event;
  }

  std::shared_ptr<Result> toResult(const itrace_t* activity) {
    auto e = lookup(activity);

    // Until we are very sure that we can reassociate kineto and profiler
    // events we need to be very defensive.
    const auto type = activity->type();
    if (e == nullptr && (type == libkineto::ActivityType::CPU_OP ||
                         type == libkineto::ActivityType::CPU_INSTANT_EVENT ||
                         type == libkineto::ActivityType::USER_ANNOTATION ||
                         type == libkineto::ActivityType::PYTHON_FUNCTION)) {
      TORCH_WARN_ONCE(
          "Detected an event which was likely passed to kineto by the PyTorch "
          "profiler, but is not present in the set of known events: ",
          activity->name(),
          " This most likely means that Kineto has not "
          "maintained address stability for this event. Please report this to "
          "the PyTorch team.");
      return nullptr;
    }

    if (e == nullptr) {
      e = resultFromActivity(activity);
      results_.get().push_back(e);
      kineto_events_[activity] = e;
    }
    return e;
  }

  void extractEventsFromTrace() {
    for (const auto* activity : trace_activities_) {
      auto e = toResult(activity);
      const auto* linked_activity = activity->linkedActivity();
      if (e && linked_activity) {
        e->visit(c10::overloaded(
            [&](ExtraFields<torch::profiler::impl::EventType::Kineto>& i) {
              i.linked_activity_ = toResult(linked_activity);
            },
            [](auto&) { TORCH_INTERNAL_ASSERT(false); }));
      }
    }
  }

  void setKinetoTID(std::shared_ptr<Result>& r,
                    std::shared_ptr<Result> parent) {
    r->visit(c10::overloaded(
        [&](ExtraFields<torch::profiler::impl::EventType::Kineto>& i) {
          TORCH_INTERNAL_ASSERT(r->start_tid_ == noTID);
          r->start_tid_ = parent ? parent->start_tid_
                                 : at::RecordFunction::currentThreadId();
        },
        [](auto&) {}));

    for (auto& child : r->children_) {
      setKinetoTID(child, r);
    }
  }

  void setParents() {
    // First pass: Collect start events and set parent to linked event.
    ska::flat_hash_map<int, std::shared_ptr<Result>> flow_map;
    for (auto& e : results_.get()) {
      TORCH_INTERNAL_ASSERT(e != nullptr);
      e->visit(c10::overloaded(
          [&](const ExtraFields<torch::profiler::impl::EventType::Kineto>& i) {
            if (i.flow.type == libkineto::kLinkAsyncCpuGpu && i.flow.start) {
              auto inserted = flow_map.insert({i.flow.id, e});
              TORCH_INTERNAL_ASSERT(inserted.second);
            }
            TORCH_INTERNAL_ASSERT(e->parent_.expired());
            e->parent_ = i.linked_activity_;
          },
          [](const auto&) {}));
    }

    // Second pass
    for (auto& e : results_.get()) {
      e->visit(c10::overloaded(
          [&](const ExtraFields<torch::profiler::impl::EventType::Kineto>& i) {
            // Flow takes priority over linked event.
            const auto it = flow_map.find(static_cast<int>(i.flow.id));
            if (it != flow_map.end() &&
                i.flow.type == libkineto::kLinkAsyncCpuGpu &&
                (i.flow.start == 0U)) {
              e->parent_ = it->second;
            }

            // If a parent was set we have to do some bookkeeping.
            auto parent = e->parent_.lock();
            if (parent) {
              parent->children_.push_back(e);
              mark_finished(e);
            }
          },
          [](const auto&) {}));
    }

    // Set TIDs now that we have established lineage.
    for (auto& e : results_.get()) {
      if (e->parent_.expired()) {
        setKinetoTID(e, nullptr);
      }
    }
  }

  static constexpr int64_t unmatchedIndex = -1;
  static constexpr auto noTID = std::numeric_limits<uint64_t>::max();
  std::reference_wrapper<std::vector<std::shared_ptr<Result>>> results_;
  std::vector<const itrace_t*> trace_activities_;
  ska::flat_hash_map<const itrace_t*, std::shared_ptr<Result>> kineto_events_;
};

ActivityTraceWrapper stopTrace() {
  return ActivityTraceWrapper{libkineto::api().activityProfiler().stopTrace()};
}

trace_ptr_t addKinetoEvents(std::vector<std::shared_ptr<Result>>& results,
                            uint64_t start_time_us, uint64_t end_time_us,
                            const ProfilerConfig& config) {
  passEventsToKineto(results, start_time_us, end_time_us);

  // In on demand mode kineto is directly controlled by other machinery.
  if (config.global()) {
    return nullptr;
  }

  auto trace = std::make_unique<ActivityTraceWrapper>(stopTrace());
  TORCH_INTERNAL_ASSERT(trace || !kKinetoAvailable);
  TransferEvents transfer{results, trace};
  return trace;
}

struct ResultGreater {
  bool operator()(const result_ptr_t& a, const result_ptr_t& b) const {
    return a->endTimeNS() > b->endTimeNS();
  }
};

void set_in_tree_building(std::vector<result_ptr_t>& results,
                          const bool value) {
  for (result_ptr_t& r : results) {
    r->visit(c10::overloaded(
        [value](ExtraFields<torch::profiler::impl::EventType::Vulkan>& i) {
          i.in_tree_building_ = value;
        },
        [&](auto&) {
          // pass
        }));
  }
}

void push_event(std::shared_ptr<Result>& event,
                ska::flat_hash_map<uint64_t, std::shared_ptr<Result>>& stacks,
                std::priority_queue<result_ptr_t, std::vector<result_ptr_t>,
                                    ResultGreater>& end_events_) {
  // Kineto builds subtrees using correlation ids and flows, so some Kineto
  // events are already marked finished before the main tree building
  // algorithm. It's fine to ignore them; the root event of these subtrees
  // not a Kineto op and will be handled normally.
  using op_fields = ExtraFields<torch::profiler::impl::EventType::TorchOp>;

  if (c10::holds_alternative<
          ExtraFields<torch::profiler::impl::EventType::Kineto>>(
          event->extra_fields_) &&
      event->finished_) {
    return;
  }

  TORCH_INTERNAL_ASSERT(event->parent_.expired());
  for (const auto& child : event->children_) {
    TORCH_INTERNAL_ASSERT(child->finished_);
  }
  TORCH_INTERNAL_ASSERT(!event->finished_);

  auto parent_it = stacks.find(event->start_tid_);
  if (parent_it == stacks.end()) {
    auto fwd_tid = event->visit(
        c10::overloaded([](const op_fields& i) { return i.forward_tid_; },
                        [](const auto&) -> uint64_t { return 0; }));
    if (fwd_tid) {
      parent_it = stacks.find(fwd_tid);
    }
  } else {
    event->parent_ = parent_it->second;
    parent_it->second->children_.push_back(event);
  }

  if (event->endTimeNS() > event->start_time_ns_) {
    stacks[event->start_tid_] = event;
    end_events_.push(event);
  } else if (event->endTimeNS() == std::numeric_limits<time_t>::min()) {
    // We use min time to indicate the lack of a termination event, so if we
    // encounter such a case we don't push to `end_events_`.
    stacks[event->start_tid_] = event;
  } else {
    mark_finished(event);
  }
}

void build_tree(std::vector<std::shared_ptr<Result>>& sorted_events) {
  set_in_tree_building(sorted_events, true);

  ska::flat_hash_map<uint64_t, std::shared_ptr<Result>> stacks;
  std::priority_queue<result_ptr_t, std::vector<result_ptr_t>, ResultGreater>
      end_events_;

  auto pop_event = [&stacks](std::shared_ptr<Result> event) {
    if (event->finished_) {
      // This event was marked finished by a previous `pop_event` call.
      return;
    }

    auto start_tid = event->start_tid_;
    auto frame = stacks.at(start_tid);

    while (frame.get() != event.get()) {
      TORCH_INTERNAL_ASSERT(frame != nullptr);
      mark_finished(frame);
      TORCH_INTERNAL_ASSERT(!frame->parent_.expired());
      frame = frame->parent_.lock();
    }

    mark_finished(event);
    stacks.erase(start_tid);
    auto new_frame = event->parent_.lock();
    if (new_frame != nullptr) {
      stacks[start_tid] = new_frame;
    }
  };

  // Stack replay loop.
  for (auto& event : sorted_events) {
    while (!end_events_.empty() &&
           end_events_.top()->endTimeNS() < event->start_time_ns_) {
      pop_event(end_events_.top());
      end_events_.pop();
    }
    push_event(event, stacks, end_events_);
  }

  // Cleanup remaining exit events.
  while (!end_events_.empty()) {
    pop_event(end_events_.top());
    end_events_.pop();
  }

  set_in_tree_building(sorted_events, false);
}

/**
 * Adjust r's duration to be the max of its current duration and the sum of all
 * of its children's adjusted durations (keeping its start time the same)
 * (adjust all child durations recursively)
 */
int64_t adjust_durations_dfs(std::shared_ptr<Result>& r) {
  if (SOFT_ASSERT(r != nullptr)) {
    int64_t original_duration = r->endTimeNS() - r->start_time_ns_;
    int64_t children_total_duration =
        std::accumulate(r->children_.begin(), r->children_.end(), 0,
                        [](int64_t acc, std::shared_ptr<Result>& child) {
                          return acc + adjust_durations_dfs(child);
                        });

    if (children_total_duration > original_duration) {
      r->visit(c10::overloaded(
          [&r, &children_total_duration](
              ExtraFields<torch::profiler::impl::EventType::TorchOp>& i) {
            i.end_time_ns_ = r->start_time_ns_ + children_total_duration;
          },
          [&children_total_duration](
              ExtraFields<torch::profiler::impl::EventType::Vulkan>& i) {
            i.duration_ns_ = children_total_duration;
          },
          [](ExtraFields<torch::profiler::impl::EventType::Allocation>& _) {
            // Pass- Allocation events can't have children
          },
          [&](auto&) {
            SOFT_ASSERT(false,
                        "unexpected event type in mobile profiler "
                        "adjust_durations_dfs: ",
                        r->name());
          }));
      return children_total_duration;
    }
    return original_duration;
  }
  return 0;
}

/**
 * 1) Adjust r's start time to be [new_start_time] (also adjusting end time and
      keeping duration the same)
 * 2) Recursively adjust r's children's start times, making them line up such
      that the last one ends at the same time as r
 * 3) Return r's final end time
 */
int64_t adjust_timestamps_dfs(std::shared_ptr<Result>& r,
                              int64_t new_start_time) {
  if (SOFT_ASSERT(r != nullptr)) {
    if (r->start_time_ns_ != new_start_time) {
      // Adjust start time (keeping duration constant)
      r->visit(c10::overloaded(
          [&r, &new_start_time](
              ExtraFields<torch::profiler::impl::EventType::TorchOp>& i) {
            i.end_time_ns_ =
                new_start_time + (i.end_time_ns_ - r->start_time_ns_);
          },
          [](ExtraFields<torch::profiler::impl::EventType::Vulkan>& i) {
            // Pass- We don't need to manually adjust end time for Vulkan events
          },
          [](ExtraFields<torch::profiler::impl::EventType::Allocation>& _) {
            // Pass- No duration or end time to adjust
          },
          [&](auto&) {
            SOFT_ASSERT(false,
                        "unexpected event type in mobile profiler "
                        "adjust_timestamps_dfs: ",
                        r->name());
          }));
      r->start_time_ns_ = new_start_time;
    }
    int64_t children_total_duration = std::accumulate(
        r->children_.begin(), r->children_.end(), 0,
        [](int64_t acc, std::shared_ptr<Result>& child) {
          return acc + (child->endTimeNS() - child->start_time_ns_);
        });

    int64_t child_start_time = r->endTimeNS() - children_total_duration;
    for (std::shared_ptr<Result>& child : r->children_) {
      child_start_time = adjust_timestamps_dfs(child, child_start_time);
    }
  }
  return r->endTimeNS();
}

/**
 * Adjust timestamps and durations of nodes in [out] such that
 *  - Vulkan event timelines are synchronized with CPU event times
 *  - Parent event timelines fully contain their child timelines
 *  - No overlaps in timelines for nodes at the same depth
 */
void adjust_timestamps(std::vector<std::shared_ptr<Result>>& out) {
  if (out.empty()) {
    return;
  }

  int64_t min_start_time = out[0]->start_time_ns_;
  for (std::shared_ptr<Result>& r : out) {
    // Only begin traversal for root nodes.
    if (r->parent_.expired()) {
      adjust_durations_dfs(r);
      min_start_time = adjust_timestamps_dfs(
          r, std::max(r->tag() != torch::profiler::impl::EventType::Vulkan
                          ? r->start_time_ns_
                          : std::numeric_limits<int64_t>::min(),
                      min_start_time));
    }
  }
}
}  // namespace

std::pair<std::vector<std::shared_ptr<Result>>,
          std::unique_ptr<ActivityTraceWrapper>>
DIPURecordQueue::getRecords(std::function<time_t(approx_time_t)> time_converter,
                            uint64_t start_time_us, uint64_t end_time_us) {
  constexpr time_t ns_per_us = 1000;
  auto converter = [&](approx_time_t t) {
    return t == std::numeric_limits<approx_time_t>::min()
               ? std::numeric_limits<time_t>::min()
               : time_converter(t);
  };

  // Used as a replacement of if-constexpr (C++ 17) to implement static
  // polymorphism.
  struct {
    std::reference_wrapper<decltype(converter)> convert;
    using Event = torch::profiler::impl::EventType;
    time_t operator()(const ExtraFields<Event::OutOfMemory>& i) const {
      return convert(i.start_time_);
    }
    time_t operator()(const ExtraFields<Event::Backend>& i) const {
      return i.start_time_us_ * ns_per_us;
    }
  } start_time_of{std::ref(converter)};

  std::vector<std::shared_ptr<Result>> out;
  std::vector<CompressedEvent> python_enters;
  for (auto& subqueue_it : sub_queues_) {
    auto& queue = *subqueue_it.second;
    auto materialize = [&](auto& events) {
      for (auto& i : events) {
        auto start_time_ns = start_time_of(i);
        out.emplace_back(Result::create(
            /*start_time_ns_=*/start_time_ns,
            /*start_tid_=*/queue.tid(),
            /*kineto_info_=*/queue.kineto_info(),
            /*extra_fields_=*/std::move(i)));
      }
      events.clear();
    };

    queue.torch_ops_.materialize(out, converter, queue.tid(),
                                 queue.kineto_info());
    materialize(queue.backend_events_);
    for (auto& i : queue.allocations_) {
      out.emplace_back(Result::create(
          /*start_time_ns_=*/converter(i.start_time_),
          /*start_tid_=*/queue.tid(),
          /*kineto_info_=*/queue.kineto_info(),
          /*extra_fields_=*/
          ExtraFields<torch::profiler::impl::EventType::Allocation>(i)));
    }
    materialize(queue.ooms_);

    for (auto& i : queue.py_calls_) {
      python_enters.push_back(
          {i.first, queue.tid(), queue.kineto_info(), converter(i.second)});
    }
  }

  if (python_tracer_) {
    for (const auto& i : python_tracer_->getEvents(
             converter, python_enters,
             static_cast<time_t>(end_time_us) * ns_per_us)) {
      out.push_back(i);
    }
    python_tracer_.reset();
  }

  if (config_.experimental_config.adjust_timestamps) {
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
      return a->start_time_ns_ < b->start_time_ns_;
    });
    build_tree(out);
    adjust_timestamps(out);
    for (auto& r : out) {
      r->parent_.reset();
      // Reset these so that second build_tree can happen
      r->finished_ = false;
      r->children_.clear();
    }
  }

  auto trace = addKinetoEvents(out, start_time_us, end_time_us, config_);

  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a->start_time_ns_ < b->start_time_ns_;
  });

  if (config_.report_input_shapes && config_.profile_memory) {
    calculateUniqueTensorIDs(out);
  }

  build_tree(out);
  return {out, std::move(trace)};
}

}  // namespace profile
}  // namespace dipu
