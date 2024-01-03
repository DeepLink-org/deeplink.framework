#include "DIPUDeviceActivity.h"

#include <GenericTraceActivity.h>
#include <output_base.h>

#include <c10/util/Exception.h>

#include "CorrelationIDManager.h"
#include "profiler.h"

namespace libkineto {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DeviceActivityInterface* device_activity_singleton = nullptr;

}  // namespace libkineto

namespace dipu {
namespace profile {

using libkineto::DeviceActivityInterface;
using libkineto::GenericTraceActivity;

DIPUDeviceActivity::~DIPUDeviceActivity() {
  disableActivities(std::set<libkineto::ActivityType>());
}

DIPUDeviceActivity& DIPUDeviceActivity::instance() {
  static DIPUDeviceActivity instance;
  return instance;
}

void DIPUDeviceActivity::pushCorrelationID(
    uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {
  CorrelationIDManager::instance().pushCorrelationID(id, type);
}

void DIPUDeviceActivity::popCorrelationID(
    DeviceActivityInterface::CorrelationFlowType type) {
  CorrelationIDManager::instance().popCorrelationID(type);
}

void DIPUDeviceActivity::enableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

void DIPUDeviceActivity::disableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

void DIPUDeviceActivity::clearActivities() { abandonAllRecords(); }

int32_t DIPUDeviceActivity::processActivities(
    libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  FlushAllRecords();
  constexpr size_t kMillisecondPerSecond = 1000;
  auto records = RecordsImpl::get().getAllRecordList();
  for (const auto& record : records) {
    GenericTraceActivity act;
    act.startTime = static_cast<int64_t>(record.begin / kMillisecondPerSecond);
    act.endTime = static_cast<int64_t>(record.end / kMillisecondPerSecond);
    act.id = static_cast<int32_t>(record.opId);
    act.device = static_cast<int32_t>(record.pid);
    act.resource = static_cast<int32_t>(record.threadIdx);
    act.flow.id = record.opId;
    if (record.isKernel) {
      act.activityType = libkineto::ActivityType::CONCURRENT_KERNEL;
      act.flow.start = false;
    } else {
      act.activityType = libkineto::ActivityType::CUDA_RUNTIME;
      act.flow.start = true;
    }
    act.activityName = record.name;
    act.flow.id = record.opId;
    act.flow.type = libkineto::kLinkAsyncCpuGpu;
    auto link_cor_id = record.linkCorrelationId;
    act.linked = linked_activity(static_cast<int32_t>(link_cor_id));
    logger.handleGenericActivity(act);
  }

  std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo>
      resource_infos = RecordsImpl::get().getResourceInfo();
  for (const auto& kv : resource_infos) {
    logger.handleResourceInfo(kv.second, start_time);
  }

  return static_cast<int32_t>(records.size());
}

void DIPUDeviceActivity::startTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  if (selected_activities.find(libkineto::ActivityType::CONCURRENT_KERNEL) !=
      selected_activities.end()) {
    setProfileOpen(true);
  }
}

void DIPUDeviceActivity::stopTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  if (selected_activities.find(libkineto::ActivityType::CONCURRENT_KERNEL) !=
      selected_activities.end()) {
    setProfileOpen(false);
  }
}

void DIPUDeviceActivity::teardownContext() {}

void DIPUDeviceActivity::setMaxBufferSize(int32_t size) {}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static DeviceActivityInterface* default_device_activity =
    &DIPUDeviceActivity::instance();

void setDeviceActivity(DeviceActivityInterface* ptr) {
  if (libkineto::device_activity_singleton == nullptr) {
    libkineto::device_activity_singleton = ptr;
    return;
  }

  if (libkineto::device_activity_singleton == default_device_activity) {
    libkineto::device_activity_singleton = ptr;
  }
}

const static int32_t default_device_activity_init = []() {
  setDeviceActivity(default_device_activity);
  return 1;
}();

}  // namespace profile
}  // namespace dipu
