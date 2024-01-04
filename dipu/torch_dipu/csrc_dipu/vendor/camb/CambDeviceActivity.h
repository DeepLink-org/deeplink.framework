// Copyright (c) 2023, DeepLink.
#pragma once

#include <DeviceActivityInterface.h>
#include <GenericTraceActivity.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <unordered_map>
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
  ~CambDeviceActivity() override;
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

  void bufferRequested(uint64_t** buffer, size_t* size, size_t* maxNumRecords);
  void bufferCompleted(uint64_t* buffer, size_t size, size_t validSize);
  std::unique_ptr<CnpapiActivityBufferMap> activityBuffers();
  void handleCnpapiActivity(
      const cnpapiActivity* record, libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
      int64_t start_time, int64_t end_time);
  void handleRuntimeActivity(
      const cnpapiActivityAPI* activity, libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
      int64_t start_time, int64_t end_time);
  void handleKernelActivity(
      const cnpapiActivityKernel* activity, libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
      int64_t start_time, int64_t end_time);
  void handleMemsetActivity(
      const cnpapiActivityMemset* activity, libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
      int64_t start_time, int64_t end_time);
  void handleMemcpyActivity(
      const cnpapiActivityMemcpy* activity, libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
      int64_t start_time, int64_t end_time);
  void handleMemcpyPtoPActivity(
      const cnpapiActivityMemcpyPtoP* activity,
      libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
      int64_t start_time, int64_t end_time);
  void handleCorrelationActivity(
      const cnpapiActivityExternalCorrelation* activity);
  const libkineto::ITraceActivity* linkedActivity(
      uint64_t correlationId,
      std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity);

  static void bufferRequestedTrampoline(uint64_t** buffer, size_t* size,
                                        size_t* maxNumRecords);
  static void bufferCompletedTrampoline(uint64_t* buffer, size_t size,
                                        size_t validSize);
  static bool nextActivityRecord(uint8_t* buffer, size_t valid_size,
                                 cnpapiActivity** record);
  static bool outOfRange(int64_t captureWindowStartTime,
                         int64_t captureWindowEndTime, int64_t activity_start,
                         int64_t activity_end);

  // TODO(caikun): naming?
  std::mutex mutex_;
  bool externalCorrelationEnabled_ = false;
  int maxMluBufferCount_ = 0;
  CnpapiActivityBufferMap allocatedMluTraceBuffers_;
  std::unique_ptr<CnpapiActivityBufferMap> readyMluTraceBuffers_;
  // cuda runtime id -> pytorch op id
  // cnpapi provides a mechanism for correlating mlu events to arbitrary
  // external events, e.g.operator activities from PyTorch.
  std::unordered_map<int64_t, int64_t> cpuCorrelationMap_;
  std::unordered_map<int64_t, int64_t> userCorrelationMap_;

  int64_t time_gap_ = 0;
};

}  // namespace dipu
