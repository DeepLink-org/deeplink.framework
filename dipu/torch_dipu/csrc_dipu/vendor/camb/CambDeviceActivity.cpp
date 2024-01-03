#include "CambDeviceActivity.h"

#include "csrc_dipu/profiler/DIPUDeviceActivity.h"

namespace dipu {

CambDeviceActivity::~CambDeviceActivity() {}

CambDeviceActivity& CambDeviceActivity::instance() {
  static CambDeviceActivity instance;
  return instance;
}

void CambDeviceActivity::pushCorrelationID(
    uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {}

void CambDeviceActivity::popCorrelationID(
    DeviceActivityInterface::CorrelationFlowType type) {}

void CambDeviceActivity::enableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

void CambDeviceActivity::disableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

void CambDeviceActivity::clearActivities() {}

int32_t CambDeviceActivity::processActivities(
    libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  return 0;
}

void CambDeviceActivity::startTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {}

void CambDeviceActivity::stopTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {}

void CambDeviceActivity::teardownContext() {}

void CambDeviceActivity::setMaxBufferSize(int32_t size) {}

const static int32_t camb_device_activity_init = []() {
  profile::setDeviceActivity(&CambDeviceActivity::instance());
  return 1;
}();

}  // namespace dipu
