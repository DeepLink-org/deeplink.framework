#pragma once

#include <DeviceActivityInterface.h>
#include <GenericTraceActivity.h>
#include <memory>

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
  DIPUDeviceActivity() = default;
};

}  // namespace profile
}  // namespace dipu
