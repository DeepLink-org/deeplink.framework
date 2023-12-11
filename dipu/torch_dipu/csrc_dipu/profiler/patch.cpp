#include <GenericTraceActivity.h>
#include <libkineto.h>
#include <unordered_set>

#include <c10/util/Exception.h>
#include <c10/util/overloaded.h>
#include <c10/util/variant.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/data_flow.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/perf-inl.h>
#include <torch/csrc/profiler/perf.h>
#include <torch/csrc/profiler/util.h>

namespace torch {
namespace profiler {
namespace impl {

ApproximateClockToUnixTimeConverter::ApproximateClockToUnixTimeConverter()
    : start_times_(measurePairs()) {}

ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair
ApproximateClockToUnixTimeConverter::measurePair() {
  // Take a measurement on either side to avoid an ordering bias.
  auto fast_0 = getApproximateTime();
  auto wall = std::chrono::system_clock::now();
  auto fast_1 = getApproximateTime();

  TORCH_INTERNAL_ASSERT(fast_1 >= fast_0, "getCount is non-monotonic.");
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
      wall.time_since_epoch());

  // `x + (y - x) / 2` is a more numerically stable average than `(x + y) / 2`.
  return {t.count(), fast_0 + (fast_1 - fast_0) / 2};
}

ApproximateClockToUnixTimeConverter::time_pairs
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
ApproximateClockToUnixTimeConverter::measurePairs() {
  static constexpr auto n_warmup = 5;
  for (C10_UNUSED const auto _ : c10::irange(n_warmup)) {
    getApproximateTime();
    steady_clock_t::now();
  }

  time_pairs out;
  for (const auto i : c10::irange(out.size())) {
    out[i] = measurePair();
  }
  return out;
}

std::function<time_t(approx_time_t)>
ApproximateClockToUnixTimeConverter::makeConverter() {
  auto end_times = measurePairs();

  // Compute the real time that passes for each tick of the approximate clock.
  std::array<long double, replicates> scale_factors{};
  for (const auto i : c10::irange(replicates)) {
    auto delta_ns = end_times[i].t_ - start_times_[i].t_;
    auto delta_approx = end_times[i].approx_t_ - start_times_[i].approx_t_;
    scale_factors[i] = static_cast<double>(delta_ns) / static_cast<double>(delta_approx);
  }
  std::sort(scale_factors.begin(), scale_factors.end());
  long double scale_factor = scale_factors[replicates / 2 + 1];

  // We shift all times by `t0` for better numerics. Double precision only has
  // 16 decimal digits of accuracy, so if we blindly multiply times by
  // `scale_factor` we may suffer from precision loss. The choice of `t0` is
  // mostly arbitrary; we just need a factor that is the correct order of
  // magnitude to bring the intermediate values closer to zero. We are not,
  // however, guaranteed that `t0_approx` is *exactly* the getApproximateTime
  // equivilent of `t0`; it is only an estimate that we have to fine tune.
  auto t0 = start_times_[0].t_;
  auto t0_approx = start_times_[0].approx_t_;
  std::array<double, replicates> t0_correction{};
  for (const auto i : c10::irange(replicates)) {
    auto dt = start_times_[i].t_ - t0;
    auto dt_approx =
        static_cast<double>(start_times_[i].approx_t_ - t0_approx) * scale_factor;
    t0_correction[i] = static_cast<double>(dt - static_cast<time_t>(dt_approx));
  }
  t0 += static_cast<time_t>(t0_correction[t0_correction.size() / 2 + 1]);

  return [=](approx_time_t t_approx) {
    // See above for why this is more stable than `A * t_approx + B`.
    auto result = static_cast<time_t>(static_cast<double>(t_approx - t0_approx) * scale_factor) + t0;
    return result;
  };
}

namespace linux_perf {

/*
 * PerfEvent
 * ---------
 */

/*
 * Syscall wrapper for perf_event_open(2)
 */
inline int64_t perf_event_open(struct perf_event_attr* hw_event, pid_t pid,
                            int cpu, int group_fd, uint64_t flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// TODO(caikun-pjlab): sync with Kineto level abstract events in profiler/events.h
static const std::unordered_map<
    std::string, std::pair<perf_type_id, /* perf event type */ uint32_t>>
    EventTable{{"cycles",
                std::make_pair(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES)},
               {"instructions",
                std::make_pair(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)},

               // Non Standard events for testing
               {"pagefaults",
                std::make_pair(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS)},
               {"backend-stall-cycles",
                std::make_pair(PERF_TYPE_HARDWARE,
                               PERF_COUNT_HW_STALLED_CYCLES_BACKEND)},
               {"frontend-stall-cycles",
                std::make_pair(PERF_TYPE_HARDWARE,
                               PERF_COUNT_HW_STALLED_CYCLES_FRONTEND)}};

PerfEvent::~PerfEvent() {
  if (fd_ > -1) {
    close(fd_);
  }
  fd_ = -1;  // poison
}

void PerfEvent::Init() {
  TORCH_CHECK(!name_.empty(), "Invalid profiler event name");

  auto const it = EventTable.find(name_);
  if (it == EventTable.end()) {
    TORCH_CHECK(false, "Unsupported profiler event name: ", name_);
  }

  struct perf_event_attr attr {};
  memset(&attr, 0, sizeof(attr));

  attr.size = sizeof(perf_event_attr);
  attr.type = it->second.first;
  attr.config = it->second.second;
  attr.disabled = 1;
  attr.inherit = 1;
  attr.exclude_kernel = 1;  // TBD
  attr.exclude_hv = 1;
  /*
   * These can be used to calculate estimated totals if the PMU is overcommitted
   * and multiplexing is happening
   */
  attr.read_format =
      PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

  pid_t pid = getpid();  // this pid
  int cpu = -1;          // all cpus
  int group_fd = -1;
  uint64_t flags = 0;

  fd_ = static_cast<int>(perf_event_open(&attr, pid, cpu, group_fd, flags));
  if (fd_ == -1) {
    TORCH_CHECK(false,
                "perf_event_open() failed, error: ", std::strerror(errno));
  }
  Reset();
}

uint64_t PerfEvent::ReadCounter() const {
  PerfCounter counter{};
  int64_t n = read(fd_, &counter, sizeof(PerfCounter));
  TORCH_CHECK(n == sizeof(counter),
              "Read failed for Perf event fd, event : ", name_,
              ", error: ", std::strerror(errno));
  TORCH_CHECK(
      counter.time_enabled == counter.time_running,
      "Hardware performance counter time multiplexing is not handled yet",
      ", name: ", name_, ", enabled: ", counter.time_enabled,
      ", running: ", counter.time_running);
  return counter.value;
}

/*
 * PerfProfiler
 * ------------
 */

void PerfProfiler::Configure(std::vector<std::string>& event_names) {
  TORCH_CHECK(event_names.size() <= MAX_EVENTS,
              "Too many events to configure, configured: ", event_names.size(),
              ", max allowed:", MAX_EVENTS);
  std::unordered_set<std::string> s(event_names.begin(), event_names.end());
  TORCH_CHECK(s.size() == event_names.size(),
              "Duplicate event names are not allowed!")
  for (auto name : event_names) {
    events_.emplace_back(name);
    events_.back().Init();
  }

  // TODO(caikun-pjlab):
  // Reset pthreadpool here to make sure we can attach to new children
  // threads
}

void PerfProfiler::Enable() {
  if (!start_values_.empty()) {
    StopCounting();
  }

  start_values_.emplace(events_.size(), 0);

  auto& sv = start_values_.top();
  for (int i = 0; i < events_.size(); ++i) {
    sv[i] = events_[i].ReadCounter();
  }
  StartCounting();
}

void PerfProfiler::Disable(perf_counters_t& vals) {
  StopCounting();
  TORCH_CHECK(vals.size() == events_.size(),
              "Can not fit all perf counters in the supplied container");
  TORCH_CHECK(!start_values_.empty(),
              "PerfProfiler must be enabled before disabling");

  /* Always connecting this disable event to the last enable event i.e. using
   * whatever is on the top of the start counter value stack. */
  perf_counters_t& sv = start_values_.top();
  for (int i = 0; i < events_.size(); ++i) {
    vals[i] = CalcDelta(sv[i], events_[i].ReadCounter());
  }
  start_values_.pop();

  // Restore it for a parent
  if (!start_values_.empty()) {
    StartCounting();
  }
}

}  // namespace linux_perf

namespace kineto {

TraceWrapper::TraceWrapper(const int64_t start_time, const std::string& name)
#ifdef USE_KINETO
    : cpu_trace_(std::make_unique<libkineto::CpuTraceBuffer>()) {
  cpu_trace_->span.startTime = start_time;
  cpu_trace_->gpuOpCount = -1;
  cpu_trace_->span.name = name;
}
#else
{
}
#endif  // USE_KINETO

TraceWrapper::~TraceWrapper() = default;

activity_t* TraceWrapper::addCPUActivity(
    const std::string& name, const libkineto::ActivityType type,
    const DeviceAndResource device_and_resource, const uint64_t correlation_id,
    const int64_t start_time, const int64_t end_time) {
#ifdef USE_KINETO
  TORCH_CHECK((bool)(*this), "Cannot add event to non-existent trace.");
  cpu_trace_->emplace_activity(cpu_trace_->span, type, name);
  auto& act = libkineto::CpuTraceBuffer::toRef(cpu_trace_->activities.back());
  act.device = device_and_resource.device;
  act.resource = device_and_resource.resource;
  act.id = static_cast<int32_t>(correlation_id);
  act.startTime = start_time;
  if (type != libkineto::ActivityType::CPU_INSTANT_EVENT) {
    act.endTime = end_time;
  }
  return cpu_trace_->activities.back().get();
#else
  return nullptr;
#endif  // USE_KINETO
}

void TraceWrapper::transferCpuTrace(int64_t end_time) {
#ifdef USE_KINETO
  cpu_trace_->span.endTime = end_time;
  libkineto::api().activityProfiler().transferCpuTrace(std::move(cpu_trace_));
#endif  // USE_KINETO
}

TraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return cpu_trace_ != nullptr;
#else
  return false;
#endif  // USE_KINETO
}

ActivityTraceWrapper::ActivityTraceWrapper(
    std::unique_ptr<interface_trace_t>&& trace)
    : trace_(std::move(trace)) {}

ActivityTraceWrapper::operator bool() const {
#ifdef USE_KINETO
  return trace_ != nullptr;
#else
  return false;
#endif  // USE_KINETO
}

void ActivityTraceWrapper::save(const std::string& path) {
#ifdef USE_KINETO
  TORCH_CHECK(!saved_, "Trace is already saved.");
  TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
#else
  TORCH_CHECK(false,
              "Saving a trace requires using torch.profiler with Kineto "
              "support (USE_KINETO=1)");
#endif  // USE_KINETO
}

void addMetadata(const activity_t* activity, const std::string& key,
                 const std::string& value) {
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const_cast<activity_t*>(activity)->addMetadata(key, value);
}

// NOLINTNEXTLINE(readability-const-return-type)
const DeviceAndResource kineto_ids() {
#ifdef USE_KINETO
  return {/*device=*/libkineto::processId(),
          /*resource=*/libkineto::systemThreadId()};
#else
  return {};
#endif  // USE_KINETO
}

const struct RegisterLibKinetoClient {
  RegisterLibKinetoClient() { libkineto::api(); }
} register_libkineto_client;

}  // namespace kineto

namespace {
constexpr TensorImplAddress NoTensorImpl{nullptr};

struct RawTensorInfo {
  TensorImplAddress impl_;
  StorageImplData storage_;
  c10::Device device_;
  bool is_free_;

  // Used to assign back to the original structs.
  std::reference_wrapper<c10::optional<AllocationID>> allocation_id_ref_;
  std::reference_wrapper<c10::optional<TensorID>> id_ref_;
};

struct RawTensors {
  std::vector<RawTensorInfo>& get() { return tensors_; }

  void operator()(TensorMetadata& t) {
    tensors_.emplace_back(RawTensorInfo{t.impl(), t.data_, t.device_, false,
                                        t.allocation_id_, t.id_});
  }

  void operator()(c10::optional<TensorMetadata>& t) {
    if (t.has_value()) {
      (*this)(*t);
    }
  }

  void operator()(ExtraFields<EventType::Allocation>& a) {
    const StorageImplData ptr{a.ptr_};
    const auto is_free = a.alloc_size_ < 0;
    tensors_.emplace_back(RawTensorInfo{NoTensorImpl, ptr, a.device(), is_free,
                                        a.allocation_id_, a.id_});
  }

  void operator()(std::vector<TensorMetadata>& t) {
    for (auto& ti : t) {
      (*this)(ti);
    }
  }

  template <typename T>
  void operator()(T& t) {}

  std::vector<RawTensorInfo> tensors_;
};

void FlattenToUniformRepresentation(std::vector<std::shared_ptr<Result>>& sorted_results,
     std::vector<RawTensorInfo>& tensors) {

  RawTensors raw_tensors;
  // The python tracer caches values, so it's only safe to use the first case.
  ska::flat_hash_set<PyModuleSelf> seen_modules;
  ska::flat_hash_set<PyOptimizerSelf> seen_optimizers;
  for (auto& result : sorted_results) {
    result->visit(c10::overloaded(
        [&](ExtraFields<EventType::TorchOp>& torch_op) {
          for (auto& i : torch_op.inputs_) {
            c10::visit(raw_tensors, i);
          }
        },
        [&](ExtraFields<EventType::PyCall>& py_call) {
          // torch.nn.Module
          if (py_call.module_.has_value() &&
              seen_modules.insert(py_call.module_->self_).second) {
            for (auto& p : py_call.module_->parameters_) {
              raw_tensors(p.metadata_);
              raw_tensors(p.grad_metadata_);
            }
          }

          // torch.optim.Optimizer
          if (py_call.optimizer_.has_value() &&
              seen_optimizers.insert(py_call.optimizer_->self_).second) {
            for (auto& p : py_call.optimizer_->parameters_) {
              raw_tensors(p.metadata_);
              raw_tensors(p.grad_metadata_);
              for (auto& state_i : p.state_) {
                raw_tensors(state_i.second);
              }
            }
          }
        },
        [&](auto& i) { raw_tensors(i); }));
  }
  tensors = std::move(raw_tensors.tensors_);
}
}  // namespace

void calculateUniqueTensorIDs(
    std::vector<std::shared_ptr<Result>>& sorted_results) {
  // This task is equivilent to https://leetcode.com/problems/number-of-islands/
  // We first cluster events with a greedy index assignment, and then merge
  // groups that overlap.
  std::vector<RawTensorInfo> tensors;
 
  // Flatten results to a uniform representation.
  // --------------------------------------------------------------------------
  FlattenToUniformRepresentation(sorted_results, tensors);
  
  // Assign IDs to solve ABA for Storage.
  // --------------------------------------------------------------------------
  {
    size_t counter{1};
    using key_t = std::pair<StorageImplData, c10::Device>;
    ska::flat_hash_map<key_t, size_t, HashCombine> versions;
    for (auto& t : tensors) {
      auto inserted = versions.insert({{t.storage_, t.device_}, counter});
      counter += static_cast<size_t>(inserted.second);
      t.allocation_id_ref_.get().emplace(AllocationID(inserted.first->second));
      if (t.is_free_) {
        versions.erase(inserted.first);
      }
    }
  }

  // Handle any allocation events which we cannot prove are for Tensor storage.
  // --------------------------------------------------------------------------
  {
    ska::flat_hash_set<AllocationID> tensor_set;
    for (const auto& t : tensors) {
      if (t.impl_ != NoTensorImpl) {
        tensor_set.insert(*t.allocation_id_ref_.get());
      }
    }
    tensors.erase(
        std::remove_if(tensors.begin(), tensors.end(),
                       [&tensor_set](const auto& i) {
                         auto it = tensor_set.find(*i.allocation_id_ref_.get());
                         return it == tensor_set.end();
                       }),
        tensors.end());
  }

  // Handle the case that the storage of a TensorImpl changed.
  // --------------------------------------------------------------------------
  using storage_id_pair_t = std::pair<AllocationID, AllocationID>;
  ska::flat_hash_set<storage_id_pair_t, HashCombine> same_group_set;
  {
    ska::flat_hash_map<TensorImplAddress, AllocationID> impl_map;
    for (const auto& t : tensors) {
      // Storage allocations / frees don't have an associated TensorImpl, so
      // we don't want all storages to merge through nullptr.
      if (!t.impl_) {
        continue;
      }

      const auto allocation_id = *t.allocation_id_ref_.get();
      const auto it = impl_map.insert({t.impl_, allocation_id}).first;

      // The pair needs to be sorted for the coalesce step to work properly.
      it->second < allocation_id
          ? same_group_set.insert({it->second, allocation_id})
          : same_group_set.insert({allocation_id, it->second});
    }
  }

  // Coalesce groups and assign final IDs.
  // --------------------------------------------------------------------------
  ska::flat_hash_map<AllocationID, size_t> id_map;
  {
    std::vector<storage_id_pair_t> unique_pairs;
    for (const auto& i : same_group_set) {
      unique_pairs.push_back(i);
    }
    std::sort(unique_pairs.begin(), unique_pairs.end());

    size_t current_id{0};
    for (const auto& i : unique_pairs) {
      auto inserted = id_map.insert({i.first, current_id});
      current_id += static_cast<size_t>(inserted.second);
      id_map.insert({i.second, inserted.first->second});
    }
  }

  // Write back to Tensor IDs.
  // --------------------------------------------------------------------------
  for (const auto& t : tensors) {
    const auto id = id_map.at(*t.allocation_id_ref_.get());
    t.id_ref_.get().emplace(TensorID(id));
  }
}

}  // namespace impl
}  // namespace profiler
}  // namespace torch
