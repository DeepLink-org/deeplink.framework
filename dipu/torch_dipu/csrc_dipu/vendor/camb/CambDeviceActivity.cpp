// Copyright (c) 2023, DeepLink.
#include "CambDeviceActivity.h"

#include <chrono>
#include <cstdlib>
#include <output_base.h>
#include <time_since_epoch.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <c10/util/Exception.h>

#include "csrc_dipu/profiler/DIPUDeviceActivity.h"
#include "csrc_dipu/utils/Log.h"
#include "csrc_dipu/vendor/camb/defines.h"

namespace dipu {

using libkineto::GenericTraceActivity;

static constexpr int32_t kBufSize(2 * 1024 * 1024);
static constexpr int64_t kNanosecondPerMicroSecond(1000);
static const std::unordered_set<int32_t> kCnrtBlackCallbackIds{
    30, 103, 113, 215, 227, 228, 236};
static const std::unordered_set<int32_t> kCndrvBlackCallbackIds{
    4, 5, 15, 61, 65, 68, 86, 89, 97};

static const std::unordered_map<libkineto::ActivityType,
                                std::vector<cnpapiActivityType>>
    kActivityTypeMapping{
        {libkineto::ActivityType::GPU_MEMCPY, {CNPAPI_ACTIVITY_TYPE_MEMCPY}},
        {libkineto::ActivityType::GPU_MEMSET, {CNPAPI_ACTIVITY_TYPE_MEMSET}},
        {libkineto::ActivityType::CONCURRENT_KERNEL,
         {CNPAPI_ACTIVITY_TYPE_KERNEL}},
        {libkineto::ActivityType::EXTERNAL_CORRELATION,
         {CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION}},
        {libkineto::ActivityType::CUDA_RUNTIME,
         {CNPAPI_ACTIVITY_TYPE_CNNL_API, CNPAPI_ACTIVITY_TYPE_KERNEL,
          CNPAPI_ACTIVITY_TYPE_MEMCPY, CNPAPI_ACTIVITY_TYPE_MEMSET,
          CNPAPI_ACTIVITY_TYPE_CNDRV_API, CNPAPI_ACTIVITY_TYPE_CNCL_API,
          CNPAPI_ACTIVITY_TYPE_RESERVED_3}},
        {libkineto::ActivityType::OVERHEAD, {CNPAPI_ACTIVITY_TYPE_OVERHEAD}}};

CambDeviceActivity::CambDeviceActivity() {
  uint64_t t0 = cnpapiGetTimestamp();
  auto wall = std::chrono::system_clock::now();
  uint64_t t1 = cnpapiGetTimestamp();
  int64_t time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         wall.time_since_epoch())
                         .count();
  time_gap_ = time_cpu - static_cast<int64_t>((t0 + t1) / 2);
}

CambDeviceActivity& CambDeviceActivity::instance() {
  static CambDeviceActivity instance;
  return instance;
}

void CambDeviceActivity::pushCorrelationID(
    uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {
  if (!external_correlation_enable_) {
    return;
  }

  switch (type) {
    case Default:
      DIPU_CALLCNPAPI(cnpapiActivityPushExternalCorrelationId(
          CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, id));
      break;
    case User:
      DIPU_CALLCNPAPI(cnpapiActivityPushExternalCorrelationId(
          CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, id));
      break;
    default:
      TORCH_CHECK(false, "unexpect correlation flow type, type: ", type);
      break;
  }
}

void CambDeviceActivity::popCorrelationID(
    DeviceActivityInterface::CorrelationFlowType type) {
  if (!external_correlation_enable_) {
    return;
  }

  switch (type) {
    case Default:
      DIPU_CALLCNPAPI(cnpapiActivityPopExternalCorrelationId(
          CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0, nullptr));
      break;
    case User:
      DIPU_CALLCNPAPI(cnpapiActivityPopExternalCorrelationId(
          CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1, nullptr));
      break;
    default:
      TORCH_CHECK(false, "unexpect correlation flow type, type: ", type);
      break;
  }
}

void CambDeviceActivity::enableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {
  if (!cnpapi_inited_) {
    DIPU_CALLCNPAPI(cnpapiInit());
    cnpapi_inited_ = true;
  }

  DIPU_CALLCNPAPI(cnpapiActivityRegisterCallbacks(bufferRequestedTrampoline,
                                                  bufferCompletedTrampoline));

  external_correlation_enable_ = false;
  for (const auto& activity_type : selected_activities) {
    const auto& iter = kActivityTypeMapping.find(activity_type);
    if (iter != kActivityTypeMapping.end()) {
      for (const auto& type : iter->second) {
        DIPU_CALLCNPAPI(cnpapiActivityEnable(type));
      }
      if (activity_type == libkineto::ActivityType::EXTERNAL_CORRELATION) {
        external_correlation_enable_ = true;
      }
    }
  }
  stopCollection = false;
}

void CambDeviceActivity::disableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {
  for (const auto& activity_type : selected_activities) {
    const auto& iter = kActivityTypeMapping.find(activity_type);
    if (iter != kActivityTypeMapping.end()) {
      for (const auto& type : iter->second) {
        DIPU_CALLCNPAPI(cnpapiActivityDisable(type));
      }
    }
  }

  external_correlation_enable_ = false;
  DIPU_CALLCNPAPI(cnpapiActivityRegisterCallbacks(nullptr, nullptr));
}

void CambDeviceActivity::clearActivities() {
  if (!cnpapi_inited_) {
    return;
  }

  DIPU_CALLCNPAPI(cnpapiActivityFlushAll());
  std::lock_guard<std::mutex> guard(mutex_);
  // Throw away ready buffers as a result of above flush
  ready_trace_buffers_ = nullptr;
  cpu_correlations_.clear();
  user_correlations_.clear();
  resource_infos_.clear();
}

int32_t CambDeviceActivity::processActivities(
    libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  std::unique_ptr<CnpapiActivityBufferMap> activities = activityBuffers();
  if (activities == nullptr || activities->empty()) {
    return 0;
  }

  int32_t count = 0;
  for (auto& act : *activities) {
    auto& buffer = act.second;
    if (buffer == nullptr || buffer->data() == nullptr || buffer->size() == 0) {
      continue;
    }

    cnpapiActivity* record = nullptr;
    while (nextActivityRecord(buffer->data(), buffer->size(), &record)) {
      handleCnpapiActivity(record, logger, linked_activity, start_time,
                           end_time);
      ++count;
    }
  }

  for (const auto& kv : resource_infos_) {
    logger.handleResourceInfo(kv.second, start_time);
  }

  return count;
}
bool CambDeviceActivity::outOfRange(int64_t profile_start_time,
                                    int64_t profile_end_time,
                                    int64_t activity_start,
                                    int64_t activity_end) {
  return activity_start < profile_start_time || activity_end > profile_end_time;
}

void CambDeviceActivity::handleRuntimeActivity(
    const cnpapiActivityAPI* activity, libkineto::ActivityLogger& logger,
    const std::function<const libkineto::ITraceActivity*(int32_t)>&
        linked_activity,
    int64_t start_time, int64_t end_time) {
  // Some mlu calls that are very frequent and also not very interesting.
  // Filter these out to reduce trace size.
  if (activity->type == CNPAPI_ACTIVITY_TYPE_CNRT_API &&
      kCnrtBlackCallbackIds.find(activity->cbid) !=
          kCnrtBlackCallbackIds.end()) {
    return;
  }
  if (activity->type == CNPAPI_ACTIVITY_TYPE_CNDRV_API &&
      kCndrvBlackCallbackIds.find(activity->cbid) !=
          kCndrvBlackCallbackIds.end()) {
    return;
  }

  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 static_cast<int64_t>(activity->start + time_gap_),
                 static_cast<int64_t>(activity->end + time_gap_))) {
    return;
  }

  GenericTraceActivity result;
  result.startTime = static_cast<int64_t>((activity->start + time_gap_) /
                                          kNanosecondPerMicroSecond);
  result.endTime = static_cast<int64_t>((activity->end + time_gap_) /
                                        kNanosecondPerMicroSecond);
  result.id = static_cast<int32_t>(activity->correlation_id);
  result.device = static_cast<int32_t>(activity->process_id);
  result.resource = static_cast<int32_t>(activity->thread_id);
  result.flow.id = activity->correlation_id;
  result.flow.start = true;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::CUDA_RUNTIME;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  char* name = nullptr;
  switch (activity->type) {
    case CNPAPI_ACTIVITY_TYPE_CNDRV_API:
      DIPU_CALLCNPAPI(cnpapiGetCallbackName(CNPAPI_CB_DOMAIN_CNDRV_API,
                                            activity->cbid,
                                            const_cast<const char**>(&name)));
      break;
    case CNPAPI_ACTIVITY_TYPE_CNRT_API:
      DIPU_CALLCNPAPI(cnpapiGetCallbackName(CNPAPI_CB_DOMAIN_CNRT_API,
                                            activity->cbid,
                                            const_cast<const char**>(&name)));
      break;
    case CNPAPI_ACTIVITY_TYPE_CNML_API:
      DIPU_CALLCNPAPI(cnpapiGetCallbackName(CNPAPI_CB_DOMAIN_CNML_API,
                                            activity->cbid,
                                            const_cast<const char**>(&name)));
      break;
    case CNPAPI_ACTIVITY_TYPE_CNNL_API:
      DIPU_CALLCNPAPI(cnpapiGetCallbackName(CNPAPI_CB_DOMAIN_CNNL_API,
                                            activity->cbid,
                                            const_cast<const char**>(&name)));
      break;
    case CNPAPI_ACTIVITY_TYPE_CNCL_API:
      DIPU_CALLCNPAPI(cnpapiGetCallbackName(CNPAPI_CB_DOMAIN_CNCL_API,
                                            activity->cbid,
                                            const_cast<const char**>(&name)));
      break;
    case CNPAPI_ACTIVITY_TYPE_CNNL_EXTRA_API:
      DIPU_CALLCNPAPI(cnpapiGetCallbackName(CNPAPI_CB_DOMAIN_CNNL_EXTRA_API,
                                            activity->cbid,
                                            const_cast<const char**>(&name)));
      break;
    default:
      TORCH_CHECK(false, "unexpect activity type, type: ", activity->type);
      break;
  }
  result.activityName = name;
  logger.handleGenericActivity(result);
}

void CambDeviceActivity::handleCorrelationActivity(
    const cnpapiActivityExternalCorrelation* activity) {
  switch (activity->external_type) {
    case CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0:
      cpu_correlations_[activity->correlation_id] = activity->external_id;
      break;
    case CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1:
      user_correlations_[activity->correlation_id] = activity->external_id;
      break;
    default:
      TORCH_CHECK(false,
                  "unexpect external type, type: ", activity->external_type);
      break;
  }
}

const libkineto::ITraceActivity* CambDeviceActivity::linkedActivity(
    uint64_t correlation_id,
    const std::function<const libkineto::ITraceActivity*(int32_t)>&
        linked_activity) {
  const auto& it = cpu_correlations_.find(correlation_id);
  if (it != cpu_correlations_.end()) {
    return linked_activity(static_cast<int32_t>(it->second));
  }

  return nullptr;
}

void CambDeviceActivity::handleKernelActivity(
    const cnpapiActivityKernel* activity, libkineto::ActivityLogger& logger,
    const std::function<const libkineto::ITraceActivity*(int32_t)>&
        linked_activity,
    int64_t start_time, int64_t end_time) {
  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 static_cast<int64_t>(activity->start + time_gap_),
                 static_cast<int64_t>(activity->end + time_gap_))) {
    return;
  }

  GenericTraceActivity result;
  result.startTime = static_cast<int64_t>((activity->start + time_gap_) /
                                          kNanosecondPerMicroSecond);
  result.endTime = static_cast<int64_t>((activity->end + time_gap_) /
                                        kNanosecondPerMicroSecond);
  result.id = static_cast<int32_t>(activity->correlation_id);
  result.device = static_cast<int32_t>(activity->device_id);
  result.resource = static_cast<int32_t>(activity->queue_id);
  result.flow.id = activity->correlation_id;
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::CONCURRENT_KERNEL;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  result.activityName = activity->name;
  logger.handleGenericActivity(result);
  recordStream(activity->device_id, activity->queue_id);
}

constexpr const char* memcpyKindString(cnpapiActivityMemcpyType kind) noexcept {
  switch (kind) {
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_HTOD:
      return "HtoD";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_DTOH:
      return "DtoH";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_DTOD:
      return "DtoD";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_HTOH:
      return "HtoH";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_PTOP:
      return "PtoP";
    case CNPAPI_ACTIVITY_MEMCPY_TYPE_UNKNOWN:
      return "unknown";
    default:
      break;
  }
  return "<unknown>";
}

void CambDeviceActivity::handleMemcpyActivity(
    const cnpapiActivityMemcpy* activity, libkineto::ActivityLogger& logger,
    const std::function<const libkineto::ITraceActivity*(int32_t)>&
        linked_activity,
    int64_t start_time, int64_t end_time) {
  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 static_cast<int64_t>(activity->start + time_gap_),
                 static_cast<int64_t>(activity->end + time_gap_))) {
    return;
  }

  GenericTraceActivity result;
  result.startTime = static_cast<int64_t>((activity->start + time_gap_) /
                                          kNanosecondPerMicroSecond);
  result.endTime = static_cast<int64_t>((activity->end + time_gap_) /
                                        kNanosecondPerMicroSecond);
  result.id = static_cast<int32_t>(activity->correlation_id);
  result.device = static_cast<int32_t>(activity->device_id);
  result.resource = static_cast<int32_t>(activity->queue_id);
  result.flow.id = activity->correlation_id;
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::GPU_MEMCPY;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  std::string name("Memcpy ");
  name += memcpyKindString(activity->copy_type);
  result.activityName = name;
  logger.handleGenericActivity(result);
  recordStream(activity->device_id, activity->queue_id);
}

void CambDeviceActivity::handleMemcpyPtoPActivity(
    const cnpapiActivityMemcpyPtoP* activity, libkineto::ActivityLogger& logger,
    const std::function<const libkineto::ITraceActivity*(int32_t)>&
        linked_activity,
    int64_t start_time, int64_t end_time) {
  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 static_cast<int64_t>(activity->start + time_gap_),
                 static_cast<int64_t>(activity->end + time_gap_))) {
    return;
  }

  GenericTraceActivity result;
  result.startTime = static_cast<int64_t>((activity->start + time_gap_) /
                                          kNanosecondPerMicroSecond);
  result.endTime = static_cast<int64_t>((activity->end + time_gap_) /
                                        kNanosecondPerMicroSecond);
  result.id = static_cast<int32_t>(activity->correlation_id);
  result.device = static_cast<int32_t>(activity->device_id);
  result.resource = static_cast<int32_t>(activity->queue_id);
  result.flow.id = activity->correlation_id;
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::GPU_MEMCPY;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  std::string name("Memcpy ");
  name += memcpyKindString(activity->copy_type);
  result.activityName = name;
  logger.handleGenericActivity(result);
  recordStream(activity->device_id, activity->queue_id);
}

void CambDeviceActivity::handleMemsetActivity(
    const cnpapiActivityMemset* activity, libkineto::ActivityLogger& logger,
    const std::function<const libkineto::ITraceActivity*(int32_t)>&
        linked_activity,
    int64_t start_time, int64_t end_time) {
  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 static_cast<int64_t>(activity->start + time_gap_),
                 static_cast<int64_t>(activity->end + time_gap_))) {
    return;
  }

  GenericTraceActivity result;
  result.startTime = static_cast<int64_t>((activity->start + time_gap_) /
                                          kNanosecondPerMicroSecond);
  result.endTime = static_cast<int64_t>((activity->end + time_gap_) /
                                        kNanosecondPerMicroSecond);
  result.id = static_cast<int32_t>(activity->correlation_id);
  result.device = static_cast<int32_t>(activity->device_id);
  result.resource = static_cast<int32_t>(activity->queue_id);
  result.flow.id = activity->correlation_id;
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::GPU_MEMSET;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  result.activityName = "Memset";
  logger.handleGenericActivity(result);
  recordStream(activity->device_id, activity->queue_id);
}

void CambDeviceActivity::handleCnpapiActivity(
    const cnpapiActivity* record, libkineto::ActivityLogger& logger,
    const std::function<const libkineto::ITraceActivity*(int32_t)>&
        linked_activity,
    int64_t start_time, int64_t end_time) {
  switch (record->type) {
    case CNPAPI_ACTIVITY_TYPE_CNRT_API:
    case CNPAPI_ACTIVITY_TYPE_CNNL_API:
    case CNPAPI_ACTIVITY_TYPE_CNDRV_API:
    case CNPAPI_ACTIVITY_TYPE_CNCL_API:
    case CNPAPI_ACTIVITY_TYPE_CNNL_EXTRA_API:
      // when calling cnpapiActivityGetNextRecord to parse profiler data
      // generated by cnpapi, we will get record in struct cnpapiActivity which
      // contains cnpapiActivityType field. Then we need cast cnpapiActivity to
      // actual struct according to cnpapiActivityType field. All cnpapiActivity
      // related structs have cnpapiActivityType field in first place, it is
      // safety to cast.
      handleRuntimeActivity(reinterpret_cast<const cnpapiActivityAPI*>(record),
                            logger, linked_activity, start_time, end_time);
      break;
    case CNPAPI_ACTIVITY_TYPE_KERNEL:
    case CNPAPI_ACTIVITY_TYPE_RESERVED_3:
      handleKernelActivity(
          reinterpret_cast<const cnpapiActivityKernel*>(record), logger,
          linked_activity, start_time, end_time);
      break;
    case CNPAPI_ACTIVITY_TYPE_MEMCPY:
      handleMemcpyActivity(
          reinterpret_cast<const cnpapiActivityMemcpy*>(record), logger,
          linked_activity, start_time, end_time);
      break;
    case CNPAPI_ACTIVITY_TYPE_MEMCPY_PTOP:
      handleMemcpyPtoPActivity(
          reinterpret_cast<const cnpapiActivityMemcpyPtoP*>(record), logger,
          linked_activity, start_time, end_time);
      break;
    case CNPAPI_ACTIVITY_TYPE_MEMSET:
      handleMemsetActivity(
          reinterpret_cast<const cnpapiActivityMemset*>(record), logger,
          linked_activity, start_time, end_time);
      break;
    case CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION:
      handleCorrelationActivity(
          reinterpret_cast<const cnpapiActivityExternalCorrelation*>(record));
      break;
    default:
      DIPU_LOG << "Unexpected activity type: " << record->type << std::endl;
      break;
  }
}

// override pure virtual function, do nothing
void CambDeviceActivity::startTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {}

// override pure virtual function, do nothing
void CambDeviceActivity::stopTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {}

// override pure virtual function, do nothing
void CambDeviceActivity::teardownContext() {}

void CambDeviceActivity::setMaxBufferSize(int32_t size) {
  max_buffer_count_ = 1 + size / kBufSize;
}

void CambDeviceActivity::bufferRequested(uint64_t** buffer, size_t* size,
                                         size_t* max_record_num) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocated_trace_buffers_.size() >= max_buffer_count_) {
    stopCollection = true;
    DIPU_LOG << "Exceeded max MLU buffer count ("
             << allocated_trace_buffers_.size() << " > " << max_buffer_count_
             << ") - terminating tracing" << std::endl;
  }

  auto buf = std::make_unique<CnpapiActivityBuffer>(kBufSize);
  *buffer = reinterpret_cast<uint64_t*>(buf->data());
  *size = kBufSize;
  allocated_trace_buffers_[reinterpret_cast<uint8_t*>(*buffer)] =
      std::move(buf);
  *max_record_num = 0;
}

void CambDeviceActivity::bufferRequestedTrampoline(uint64_t** buffer,
                                                   size_t* size,
                                                   size_t* max_record_num) {
  instance().bufferRequested(buffer, size, max_record_num);
}

void CambDeviceActivity::bufferCompleted(uint64_t* buffer, size_t size,
                                         size_t valid_size) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocated_trace_buffers_.find(reinterpret_cast<uint8_t*>(buffer));
  TORCH_CHECK(it != allocated_trace_buffers_.end(),
              "bufferCompleted called with unknown buffer");

  if (!ready_trace_buffers_) {
    ready_trace_buffers_ = std::make_unique<CnpapiActivityBufferMap>();
  }
  // Set valid size of buffer before moving to ready map
  it->second->setSize(valid_size);
  (*ready_trace_buffers_)[it->first] = std::move(it->second);
  allocated_trace_buffers_.erase(it);
}

void CambDeviceActivity::bufferCompletedTrampoline(uint64_t* buffer,
                                                   size_t size,
                                                   size_t valid_size) {
  instance().bufferCompleted(buffer, 0, valid_size);
}

bool CambDeviceActivity::nextActivityRecord(uint8_t* buffer, size_t valid_size,
                                            cnpapiActivity** record) {
  cnpapiResult status = cnpapiActivityGetNextRecord(buffer, valid_size, record);
  if (status != CNPAPI_SUCCESS) {
    return false;
  }
  return *record != nullptr;
}

std::unique_ptr<CnpapiActivityBufferMap> CambDeviceActivity::activityBuffers() {
  DIPU_CALLCNPAPI(cnpapiActivityFlushAll());

  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(ready_trace_buffers_);
}

std::string streamName(int64_t stream_id, const std::string& postfix) {
  std::string name("stream ");
  name += std::to_string(stream_id);
  if (!postfix.empty()) {
    name += " ";
    name += postfix;
  }
  return name;
}

void CambDeviceActivity::recordStream(uint64_t device_index, uint64_t stream_id,
                                      const std::string& postfix) {
  auto device = static_cast<int64_t>(device_index);
  auto stream = static_cast<int64_t>(stream_id);
  auto iter = resource_infos_.find({device, stream});
  if (iter == resource_infos_.end()) {
    resource_infos_.emplace_hint(
        iter, std::make_pair(device, stream),
        libkineto::ResourceInfo(device, stream, stream,
                                streamName(stream, postfix)));
  }
}

const static int32_t camb_device_activity_init = []() {
  const char* env = std::getenv("FORCE_USE_DIPU_PROFILER");
  if ((env == nullptr) || (strncmp(env, "false", strlen("false")) == 0) ||
      (strncmp(env, "False", strlen("False")) == 0)) {
    libkineto::device_activity_singleton = &CambDeviceActivity::instance();
    return 1;
  }
  return 0;
}();

}  // namespace dipu
