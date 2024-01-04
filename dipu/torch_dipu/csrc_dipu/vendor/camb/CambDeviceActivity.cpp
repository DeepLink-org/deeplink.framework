// Copyright (c) 2023, DeepLink.
#include "CambDeviceActivity.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <output_base.h>
#include <time_since_epoch.h>

#include <c10/util/Exception.h>

#include "csrc_dipu/profiler/DIPUDeviceActivity.h"
#include "csrc_dipu/utils/Log.h"
#include "csrc_dipu/vendor/vendorapi.h"

namespace dipu {

using libkineto::GenericTraceActivity;

constexpr size_t kBufSize(2 * 1024 * 1024);
// TODO(caikun): rename it, papi timestamp has gap with cpu timestamp
constexpr size_t kNanosecondPerMicroSecond = 1000;

CambDeviceActivity::CambDeviceActivity() {
  // refactor timing
  // https://www.cambricon.com/docs/sdk_1.14.0/cntoolkit_3.6.1/cnpapi_3.6.0/cnpapi_api/cnpapi_api.html?highlight=cnpapigettimestamp#cnpapigettimestamp
  uint64_t cnpapitime = cnpapiGetTimestamp();
  std::cout << "1th cnpapitime = " << cnpapitime << std::endl;
  int64_t time_cpu =
      libkineto::timeSinceEpoch(std::chrono::system_clock::now());
  cnpapitime = cnpapiGetTimestamp();
  std::cout << "2th cnpapitime = " << cnpapitime << std::endl;
  // TODO(caikun): time_cpu should be micro second and time_cpu * 1000, but got
  // nano second!!!!
  time_gap_ = time_cpu * kNanosecondPerMicroSecond - cnpapitime;
  std::cout << "time_cpu = " << time_cpu << ", time_gap_ = " << time_gap_
            << std::endl;
}

CambDeviceActivity::~CambDeviceActivity() {}

CambDeviceActivity& CambDeviceActivity::instance() {
  static CambDeviceActivity instance;
  return instance;
}

void CambDeviceActivity::pushCorrelationID(
    uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {
  std::cout << "enter into pushCorrelationID, id = " << id << std::endl;
  if (!externalCorrelationEnabled_) {
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
    default:
      TORCH_CHECK(false, "unexpect correlation flow type, type: ", type);
  }
}

void CambDeviceActivity::popCorrelationID(
    DeviceActivityInterface::CorrelationFlowType type) {
  std::cout << "enter into popCorrelationID" << std::endl;
  if (!externalCorrelationEnabled_) {
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
    default:
      TORCH_CHECK(false, "unexpect correlation flow type, type: ", type);
  }
}

void CambDeviceActivity::enableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {
  std::cout << "enter into enableActivities" << std::endl;
  static bool inited = false;
  if (!inited) {
    std::cout << "call cnpapiInit" << std::endl;
    DIPU_CALLCNPAPI(cnpapiInit());
    inited = true;
  }

  std::cout << "call cnpapiActivityRegisterCallbacks" << std::endl;
  DIPU_CALLCNPAPI(cnpapiActivityRegisterCallbacks(bufferRequestedTrampoline,
                                                  bufferCompletedTrampoline));

  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == libkineto::ActivityType::GPU_MEMCPY) {
      std::cout << "call cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY)"
                << std::endl;
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
    }
    if (activity == libkineto::ActivityType::GPU_MEMSET) {
      std::cout << "call cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET)"
                << std::endl;
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    }
    if (activity == libkineto::ActivityType::CONCURRENT_KERNEL) {
      std::cout << "call cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL)"
                << std::endl;
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
    }
    if (activity == libkineto::ActivityType::EXTERNAL_CORRELATION) {
      std::cout
          << "call "
             "cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION)"
          << std::endl;
      DIPU_CALLCNPAPI(
          cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION));
      externalCorrelationEnabled_ = true;
    }
    if (activity == libkineto::ActivityType::CUDA_RUNTIME) {
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CNNL_API));
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_KERNEL));
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_MEMSET));
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_CNCL_API));
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_RESERVED_3));
      std::cout << "call cnpapiActivityEnable(CUDA_RUNTIME)" << std::endl;
    }
    if (activity == libkineto::ActivityType::OVERHEAD) {
      DIPU_CALLCNPAPI(cnpapiActivityEnable(CNPAPI_ACTIVITY_TYPE_OVERHEAD));
    }
  }
  stopCollection = false;
}

void CambDeviceActivity::disableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {
  std::cout << "enter into disableActivities" << std::endl;
  for (const auto& activity : selected_activities) {
    if (activity == libkineto::ActivityType::GPU_MEMCPY) {
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
    }
    if (activity == libkineto::ActivityType::GPU_MEMSET) {
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMSET));
    }
    if (activity == libkineto::ActivityType::CONCURRENT_KERNEL) {
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_KERNEL));
    }
    if (activity == libkineto::ActivityType::EXTERNAL_CORRELATION) {
      DIPU_CALLCNPAPI(
          cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_EXTERNAL_CORRELATION));
    }
    if (activity == libkineto::ActivityType::CUDA_RUNTIME) {
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CNNL_API));
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_KERNEL));
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMCPY));
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_MEMSET));
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CNDRV_API));
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_CNCL_API));
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_RESERVED_3));
    }
    if (activity == libkineto::ActivityType::OVERHEAD) {
      DIPU_CALLCNPAPI(cnpapiActivityDisable(CNPAPI_ACTIVITY_TYPE_OVERHEAD));
    }
  }
  externalCorrelationEnabled_ = false;
  DIPU_CALLCNPAPI(cnpapiActivityRegisterCallbacks(nullptr, nullptr));
}

void CambDeviceActivity::clearActivities() {
  std::cout << "enter into clearActivities" << std::endl;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (allocatedMluTraceBuffers_.empty()) {
      return;
    }
  }
  DIPU_CALLCNPAPI(cnpapiActivityFlushAll());
  std::lock_guard<std::mutex> guard(mutex_);
  // Throw away ready buffers as a result of above flush
  readyMluTraceBuffers_ = nullptr;
  cpuCorrelationMap_.clear();
  userCorrelationMap_.clear();
}

int32_t CambDeviceActivity::processActivities(
    libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  std::cout << "enter into processActivities" << std::endl;
  std::unique_ptr<CnpapiActivityBufferMap> activities = activityBuffers();
  if (activities == nullptr || activities->empty()) {
    std::cout << "activities is empty" << std::endl;
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

  return count;
}
bool CambDeviceActivity::outOfRange(int64_t captureWindowStartTime,
                                    int64_t captureWindowEndTime,
                                    int64_t activity_start,
                                    int64_t activity_end) {
  return activity_start < captureWindowStartTime ||
         activity_end > captureWindowEndTime;
}

void CambDeviceActivity::handleRuntimeActivity(
    const cnpapiActivityAPI* activity, libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  // Some mlu calls that are very frequent and also not very interesting.
  // Filter these out to reduce trace size.
  if (activity->type == CNPAPI_ACTIVITY_TYPE_CNRT_API) {
    if (activity->cbid == 30 || activity->cbid == 227 ||
        activity->cbid == 113 || activity->cbid == 228 ||
        activity->cbid == 103 || activity->cbid == 236 ||
        activity->cbid == 215) {
      return;
    }
  }
  if (activity->type == CNPAPI_ACTIVITY_TYPE_CNDRV_API) {
    if (activity->cbid == 86 || activity->cbid == 65 || activity->cbid == 68 ||
        activity->cbid == 61 || activity->cbid == 97 || activity->cbid == 89 ||
        activity->cbid == 15 || activity->cbid == 5 || activity->cbid == 4) {
      return;
    }
  }
  std::cout << "enter into handleRuntimeActivity, activity->type = "
            << activity->type << ", activity->cbid = " << activity->cbid
            << std::endl;

  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 activity->start + time_gap_, activity->end + time_gap_)) {
    std::cout << "record out of range, start_time=" << start_time
              << ", end_time=" << end_time
              << ", activity->start=" << activity->start
              << ", activity->end=" << activity->end
              << ", activity->start + time_gap_=" << activity->start + time_gap_
              << ", activity->end + time_gap_=" << activity->end + time_gap_
              << ", time_gap_=" << time_gap_ << std::endl;
    return;
  }

  GenericTraceActivity result;
  result.startTime = (activity->start + time_gap_) / kNanosecondPerMicroSecond;
  result.endTime = (activity->end + time_gap_) / kNanosecondPerMicroSecond;
  result.id = activity->correlation_id;
  result.device = activity->process_id;
  result.resource = activity->thread_id;
  result.flow.id = activity->correlation_id;
  // TODO(caikun): is it right? only cudaLaunchKernel be true?
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
  std::cout << "enter into handleRuntimeActivity, correlation_id="
            << activity->correlation_id << ", name=" << name << std::endl;
  logger.handleGenericActivity(result);
}

void CambDeviceActivity::handleCorrelationActivity(
    const cnpapiActivityExternalCorrelation* activity) {
  std::cout << "enter into handleCorrelationActivity, correlation_id="
            << activity->correlation_id
            << ", external_id=" << activity->external_id << std::endl;
  switch (activity->external_type) {
    case CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM0:
      cpuCorrelationMap_[activity->correlation_id] = activity->external_id;
      break;
    case CNPAPI_EXTERNAL_CORRELATION_TYPE_CUSTOM1:
      userCorrelationMap_[activity->correlation_id] = activity->external_id;
      break;
    default:
      TORCH_CHECK(false,
                  "unexpect external type, type: ", activity->external_type);
  }
}

const libkineto::ITraceActivity* CambDeviceActivity::linkedActivity(
    uint64_t correlationId,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity) {
  const auto& it = cpuCorrelationMap_.find(correlationId);
  if (it != cpuCorrelationMap_.end()) {
    return linked_activity(static_cast<int32_t>(it->second));
  }
  std::cout << "enter into linkedActivity, not found record, correlationId="
            << correlationId << std::endl;
  return nullptr;
}

void CambDeviceActivity::handleKernelActivity(
    const cnpapiActivityKernel* activity, libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  std::cout << "enter into handleKernelActivity, activity->type = "
            << activity->type << std::endl;

  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 activity->start + time_gap_, activity->end + time_gap_)) {
    std::cout << "record out of range, start_time=" << start_time
              << ", end_time=" << end_time
              << ", activity->start=" << activity->start
              << ", activity->end=" << activity->end
              << ", activity->start + time_gap_=" << activity->start + time_gap_
              << ", activity->end + time_gap_=" << activity->end + time_gap_
              << ", time_gap_=" << time_gap_ << std::endl;
    return;
  }

  GenericTraceActivity result;
  result.startTime = (activity->start + time_gap_) / kNanosecondPerMicroSecond;
  result.endTime = (activity->end + time_gap_) / kNanosecondPerMicroSecond;
  result.id = activity->correlation_id;
  result.device = activity->device_id;
  result.resource = activity->queue_id;
  result.flow.id = activity->correlation_id;
  // TODO(caikun): is it right? only cudaLaunchKernel be true?
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::CONCURRENT_KERNEL;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  result.activityName = activity->name;
  std::cout << "enter into handleKernelActivity, correlation_id="
            << activity->correlation_id << ", activity->name=" << activity->name
            << std::endl;
  logger.handleGenericActivity(result);
}

const char* memcpyKindString(cnpapiActivityMemcpyType kind) {
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
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  std::cout << "enter into handleMemcpyActivity, activity->type = "
            << activity->type << std::endl;

  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 activity->start + time_gap_, activity->end + time_gap_)) {
    std::cout << "record out of range, start_time=" << start_time
              << ", end_time=" << end_time
              << ", activity->start=" << activity->start
              << ", activity->end=" << activity->end
              << ", activity->start + time_gap_=" << activity->start + time_gap_
              << ", activity->end + time_gap_=" << activity->end + time_gap_
              << ", time_gap_=" << time_gap_ << std::endl;
    return;
  }

  GenericTraceActivity result;
  result.startTime = (activity->start + time_gap_) / kNanosecondPerMicroSecond;
  result.endTime = (activity->end + time_gap_) / kNanosecondPerMicroSecond;
  result.id = activity->correlation_id;
  result.device = activity->device_id;
  result.resource = activity->queue_id;
  result.flow.id = activity->correlation_id;
  // TODO(caikun): is it right? only cudaLaunchKernel be true?
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::GPU_MEMCPY;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  std::string name("Memcpy ");
  name += memcpyKindString(activity->copy_type);
  result.activityName = name;
  std::cout << "enter into handleMemcpyActivity, correlation_id="
            << activity->correlation_id
            << ", result.activityName=" << result.activityName << std::endl;
  logger.handleGenericActivity(result);
}

void CambDeviceActivity::handleMemcpyPtoPActivity(
    const cnpapiActivityMemcpyPtoP* activity, libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  std::cout << "enter into handleMemcpyPtoPActivity, activity->type = "
            << activity->type << std::endl;

  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 activity->start + time_gap_, activity->end + time_gap_)) {
    std::cout << "record out of range, start_time=" << start_time
              << ", end_time=" << end_time
              << ", activity->start=" << activity->start
              << ", activity->end=" << activity->end
              << ", activity->start + time_gap_=" << activity->start + time_gap_
              << ", activity->end + time_gap_=" << activity->end + time_gap_
              << ", time_gap_=" << time_gap_ << std::endl;
    return;
  }

  GenericTraceActivity result;
  result.startTime = (activity->start + time_gap_) / kNanosecondPerMicroSecond;
  result.endTime = (activity->end + time_gap_) / kNanosecondPerMicroSecond;
  result.id = activity->correlation_id;
  result.device = activity->device_id;
  result.resource = activity->queue_id;
  result.flow.id = activity->correlation_id;
  // TODO(caikun): is it right? only cudaLaunchKernel be true?
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::GPU_MEMCPY;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  std::string name("Memcpy ");
  name += memcpyKindString(activity->copy_type);
  result.activityName = name;
  std::cout << "enter into handleMemcpyPtoPActivity, correlation_id="
            << activity->correlation_id
            << ", result.activityName=" << result.activityName << std::endl;
  logger.handleGenericActivity(result);
}

void CambDeviceActivity::handleMemsetActivity(
    const cnpapiActivityMemset* activity, libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  std::cout << "enter into handleMemsetActivity, activity->type = "
            << activity->type << std::endl;

  if (outOfRange(start_time * kNanosecondPerMicroSecond,
                 end_time * kNanosecondPerMicroSecond,
                 activity->start + time_gap_, activity->end + time_gap_)) {
    std::cout << "record out of range, start_time=" << start_time
              << ", end_time=" << end_time
              << ", activity->start=" << activity->start
              << ", activity->end=" << activity->end
              << ", activity->start + time_gap_=" << activity->start + time_gap_
              << ", activity->end + time_gap_=" << activity->end + time_gap_
              << ", time_gap_=" << time_gap_ << std::endl;
    return;
  }

  GenericTraceActivity result;
  result.startTime = (activity->start + time_gap_) / kNanosecondPerMicroSecond;
  result.endTime = (activity->end + time_gap_) / kNanosecondPerMicroSecond;
  result.id = activity->correlation_id;
  result.device = activity->device_id;
  result.resource = activity->queue_id;
  result.flow.id = activity->correlation_id;
  // TODO(caikun): is it right? only cudaLaunchKernel be true?
  result.flow.start = false;
  result.flow.type = libkineto::kLinkAsyncCpuGpu;
  result.activityType = libkineto::ActivityType::GPU_MEMSET;
  result.linked = linkedActivity(activity->correlation_id, linked_activity);
  result.activityName = "Memset";
  std::cout << "enter into handleMemsetActivity, correlation_id="
            << activity->correlation_id
            << ", result.activityName=" << result.activityName << std::endl;
  logger.handleGenericActivity(result);
}

void CambDeviceActivity::handleCnpapiActivity(
    const cnpapiActivity* record, libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  // std::cout << "enter into handleCnpapiActivity, record->type=" <<
  // record->type
  // << std::endl;
  switch (record->type) {
    case CNPAPI_ACTIVITY_TYPE_CNRT_API:
    case CNPAPI_ACTIVITY_TYPE_CNNL_API:
    case CNPAPI_ACTIVITY_TYPE_CNDRV_API:
    case CNPAPI_ACTIVITY_TYPE_CNCL_API:
    case CNPAPI_ACTIVITY_TYPE_CNNL_EXTRA_API:
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

void CambDeviceActivity::startTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  std::cout << "enter into startTrace" << std::endl;
}

void CambDeviceActivity::stopTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  std::cout << "enter into stopTrace" << std::endl;
}

void CambDeviceActivity::teardownContext() {
  std::cout << "enter into teardownContext" << std::endl;
}

void CambDeviceActivity::setMaxBufferSize(int32_t size) {
  maxMluBufferCount_ = 1 + size / kBufSize;
  std::cout << "enter into setMaxBufferSize, size = " << size
            << ", maxMluBufferCount_=" << maxMluBufferCount_ << std::endl;
}

void CambDeviceActivity::bufferRequested(uint64_t** buffer, size_t* size,
                                         size_t* maxNumRecords) {
  std::cout << "enter into bufferRequested" << std::endl;
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedMluTraceBuffers_.size() >= maxMluBufferCount_) {
    stopCollection = true;
    DIPU_LOG << "Exceeded max MLU buffer count ("
             << allocatedMluTraceBuffers_.size() << " > " << maxMluBufferCount_
             << ") - terminating tracing" << std::endl;
  }

  auto buf = std::make_unique<CnpapiActivityBuffer>(kBufSize);
  *buffer = (uint64_t*)buf->data();
  *size = kBufSize;
  allocatedMluTraceBuffers_[(uint8_t*)(*buffer)] = std::move(buf);
  *maxNumRecords = 0;
}

void CambDeviceActivity::bufferRequestedTrampoline(uint64_t** buffer,
                                                   size_t* size,
                                                   size_t* maxNumRecords) {
  std::cout << "enter into bufferRequestedTrampoline" << std::endl;
  instance().bufferRequested(buffer, size, maxNumRecords);
}

void CambDeviceActivity::bufferCompleted(uint64_t* buffer, size_t size,
                                         size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = allocatedMluTraceBuffers_.find((uint8_t*)buffer);
  TORCH_CHECK(it != allocatedMluTraceBuffers_.end(),
              "bufferCompleted called with unknown buffer");

  if (!readyMluTraceBuffers_) {
    readyMluTraceBuffers_ = std::make_unique<CnpapiActivityBufferMap>();
  }
  // Set valid size of buffer before moving to ready map
  it->second->setSize(validSize);
  (*readyMluTraceBuffers_)[it->first] = std::move(it->second);
  allocatedMluTraceBuffers_.erase(it);
  std::cout << "enter into bufferCompleted, readyMluTraceBuffers_ not empty"
            << std::endl;
}

void CambDeviceActivity::bufferCompletedTrampoline(uint64_t* buffer,
                                                   size_t size,
                                                   size_t validSize) {
  std::cout << "enter into bufferCompletedTrampoline, validSize = " << validSize
            << std::endl;
  instance().bufferCompleted(buffer, 0, validSize);
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
  // TODO(caikun): delete it!!!!
  // {
  //   std::lock_guard<std::mutex> guard(mutex_);
  //   if (allocatedMluTraceBuffers_.empty()) {
  //     std::cout
  //         << "enter into activityBuffers, allocatedMluTraceBuffers_ is empty"
  //         << std::endl;
  //     return nullptr;
  //   }
  // }

  // Can't hold mutex_ during this call, since bufferCompleted
  // will be called by libcnapi and mutex_ is acquired there.
  DIPU_CALLCNPAPI(cnpapiActivityFlushAll());

  std::lock_guard<std::mutex> guard(mutex_);
  // Transfer ownership of buffers to caller. A new map is created on-demand.
  return std::move(readyMluTraceBuffers_);
}

const static int32_t camb_device_activity_init = []() {
  const char* env = std::getenv("FORCE_USE_DIPU_PROFILER");
  if ((env == nullptr) || (strncmp(env, "false", 5) == 0) ||
      (strncmp(env, "False", 5) == 0)) {
    profile::setDeviceActivity(&CambDeviceActivity::instance());
    return 1;
  }
  return 0;
}();

}  // namespace dipu
