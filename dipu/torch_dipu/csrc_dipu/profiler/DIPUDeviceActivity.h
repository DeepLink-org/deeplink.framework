#pragma once

#include <DeviceActivityInterface.h>
#include <GenericTraceActivity.h>
#include <memory>
#include <unordered_map>

namespace dipu {
namespace profile {

class DIPUDeviceActivity : public libkineto::DeviceActivityInterface {
 public:
  ~DIPUDeviceActivity() override;
  DIPUDeviceActivity(const DIPUDeviceActivity&) = delete;
  DIPUDeviceActivity& operator=(const DIPUDeviceActivity&) = delete;

  // DIPUDeviceActivity designed as a singleton
  static DIPUDeviceActivity& instance();

  void pushCorrelationID(
      uint64_t id,
      libkineto::DeviceActivityInterface::CorrelationFlowType type) override;
  void popCorrelationID(
      libkineto::DeviceActivityInterface::CorrelationFlowType type) override;

  void enableActivities(
      const std::set<libkineto::ActivityType>& selectedActivities) override;
  void disableActivities(
      const std::set<libkineto::ActivityType>& selectedActivities) override;
  void clearActivities() override;
  int32_t processActivities(
      libkineto::ActivityLogger& logger,
      std::function<const libkineto::ITraceActivity*(int32_t)> linkedActivity,
      int64_t startTime, int64_t endTime) override;

  void teardownContext() override;
  void setMaxBufferSize(int32_t size) override;

 private:
  DIPUDeviceActivity() = default;

  std::unordered_map<uint64_t, std::unique_ptr<libkineto::GenericTraceActivity>>
      cpu_activities_;
  std::unordered_map<uint64_t, std::unique_ptr<libkineto::GenericTraceActivity>>
      device_activities_;
};

}  // namespace profile
}  // namespace dipu
