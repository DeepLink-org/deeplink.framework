// Copyright (c) 2023, DeepLink.
#pragma once

#include <DeviceActivityInterface.h>
#include <GenericTraceActivity.h>
#include <IActivityProfiler.h>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cnpapi.h>

namespace dipu {

class CnpapiActivityBuffer {
 public:
  explicit CnpapiActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  CnpapiActivityBuffer() = delete;
  CnpapiActivityBuffer& operator=(const CnpapiActivityBuffer&) = delete;
  CnpapiActivityBuffer(CnpapiActivityBuffer&&) = default;
  CnpapiActivityBuffer& operator=(CnpapiActivityBuffer&&) = default;

  size_t size() const { return size_; }

  void setSize(size_t size) {
    assert(size <= buf_.capacity());
    size_ = size;
  }

  uint8_t* data() { return buf_.data(); }

 private:
  std::vector<uint8_t> buf_;
  size_t size_;
};

using CnpapiActivityBufferMap =
    std::map<uint8_t*, std::unique_ptr<CnpapiActivityBuffer>>;

class CambDeviceActivity : public libkineto::DeviceActivityInterface {
 public:
  ~CambDeviceActivity() override = default;
  CambDeviceActivity(const CambDeviceActivity&) = delete;
  CambDeviceActivity& operator=(const CambDeviceActivity&) = delete;

  // CambDeviceActivity designed as a singleton
  static CambDeviceActivity& instance();

  void pushCorrelationID(
      uint64_t id,
      libkineto::DeviceActivityInterface::CorrelationFlowType type) override;
  void popCorrelationID(
      libkineto::DeviceActivityInterface::CorrelationFlowType type) override;

  void enableActivities(
      const std::set<libkineto::ActivityType>& selected_activities) override;
  void disableActivities(
      const std::set<libkineto::ActivityType>& selected_activities) override;
  void clearActivities() override;
  int32_t processActivities(
      libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
      int64_t start_time, int64_t end_time) override;

  void startTrace(
      const std::set<libkineto::ActivityType>& selected_activities) override;
  void stopTrace(
      const std::set<libkineto::ActivityType>& selected_activities) override;

  void teardownContext() override;
  void setMaxBufferSize(int32_t size) override;

 private:
  CambDeviceActivity();

  void bufferRequested(uint64_t** buffer, size_t* size, size_t* max_record_num);
  void bufferCompleted(uint64_t* buffer, size_t size, size_t valid_size);
  std::unique_ptr<CnpapiActivityBufferMap> activityBuffers();
  const libkineto::ITraceActivity* linkedActivity(
      uint64_t correlation_id,
      const std::function<const libkineto::ITraceActivity*(int32_t)>&
          linked_activity);
  void recordStream(uint64_t device_index, uint64_t stream_id,
                    const std::string& postfix = "");

  void handleCnpapiActivity(
      const cnpapiActivity* record, libkineto::ActivityLogger& logger,
      const std::function<const libkineto::ITraceActivity*(int32_t)>&
          linked_activity,
      int64_t start_time, int64_t end_time);
  void handleRuntimeActivity(
      const cnpapiActivityAPI* activity, libkineto::ActivityLogger& logger,
      const std::function<const libkineto::ITraceActivity*(int32_t)>&
          linked_activity,
      int64_t start_time, int64_t end_time);
  void handleKernelActivity(
      const cnpapiActivityKernel* activity, libkineto::ActivityLogger& logger,
      const std::function<const libkineto::ITraceActivity*(int32_t)>&
          linked_activity,
      int64_t start_time, int64_t end_time);
  void handleMemsetActivity(
      const cnpapiActivityMemset* activity, libkineto::ActivityLogger& logger,
      const std::function<const libkineto::ITraceActivity*(int32_t)>&
          linked_activity,
      int64_t start_time, int64_t end_time);
  void handleMemcpyActivity(
      const cnpapiActivityMemcpy* activity, libkineto::ActivityLogger& logger,
      const std::function<const libkineto::ITraceActivity*(int32_t)>&
          linked_activity,
      int64_t start_time, int64_t end_time);
  void handleMemcpyPtoPActivity(
      const cnpapiActivityMemcpyPtoP* activity,
      libkineto::ActivityLogger& logger,
      const std::function<const libkineto::ITraceActivity*(int32_t)>&
          linked_activity,
      int64_t start_time, int64_t end_time);
  void handleCorrelationActivity(
      const cnpapiActivityExternalCorrelation* activity);

  static void bufferRequestedTrampoline(uint64_t** buffer, size_t* size,
                                        size_t* max_record_num);
  static void bufferCompletedTrampoline(uint64_t* buffer, size_t size,
                                        size_t valid_size);
  static bool nextActivityRecord(uint8_t* buffer, size_t valid_size,
                                 cnpapiActivity** record);
  static bool outOfRange(int64_t profile_start_time, int64_t profile_end_time,
                         int64_t activity_start, int64_t activity_end);

  std::mutex mutex_;
  bool cnpapi_inited_ = false;
  bool external_correlation_enable_ = false;
  int32_t max_buffer_count_ = 0;
  CnpapiActivityBufferMap allocated_trace_buffers_;
  std::unique_ptr<CnpapiActivityBufferMap> ready_trace_buffers_;
  // cuda runtime id -> pytorch op id
  // cnpapi provides a mechanism for correlating mlu events to arbitrary
  // external events, e.g.operator activities from PyTorch.
  std::unordered_map<uint64_t, uint64_t> cpu_correlations_;
  std::unordered_map<uint64_t, uint64_t> user_correlations_;
  std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo>
      resource_infos_;
  int64_t time_gap_ = 0;
};

}  // namespace dipu
