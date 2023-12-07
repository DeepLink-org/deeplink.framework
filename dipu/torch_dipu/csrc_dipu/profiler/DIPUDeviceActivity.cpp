#include "DIPUDeviceActivity.h"

#include <GenericTraceActivity.h>
#include <output_base.h>

#include <c10/util/Exception.h>

#include "CorrelationIDManager.h"
#include "profiler.h"

namespace dipu {
namespace profile {

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
    const std::set<libkineto::ActivityType>& selectedActivities) {}

void DIPUDeviceActivity::disableActivities(
    const std::set<libkineto::ActivityType>& selectedActivities) {
  if (selectedActivities.find(libkineto::ActivityType::CONCURRENT_KERNEL) !=
      selectedActivities.end()) {
    setProfileOpen(false);
  }
}

void DIPUDeviceActivity::clearActivities() {
  abandonAllRecords();
  cpu_activities_.clear();
  device_activities_.clear();
}

int32_t DIPUDeviceActivity::processActivities(
    libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linkedActivity,
    int64_t startTime, int64_t endTime) {
  FlushAllRecords();
  constexpr size_t kSecondToMillisecond = 1000;
  auto records = RecordsImpl::get().getAllRecordList();
  for (const auto& record : records) {
    GenericTraceActivity act;
    act.startTime = static_cast<int64_t>(record.begin / kSecondToMillisecond);
    act.endTime = static_cast<int64_t>(record.end / kSecondToMillisecond);
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
    act.linked = linkedActivity(static_cast<int32_t>(link_cor_id));
    logger.handleGenericActivity(act);
  }

  std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo>
      resource_infos = RecordsImpl::get().getResourceInfo();
  for (const auto& kv : resource_infos) {
    logger.handleResourceInfo(kv.second, startTime);
  }

  return static_cast<int32_t>(records.size());
}

void DIPUDeviceActivity::teardownContext() {}

void DIPUDeviceActivity::setMaxBufferSize(int32_t size) {}

}  // namespace profile
}  // namespace dipu

namespace libkineto {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DeviceActivityInterface* device_activity_singleton =
    &dipu::profile::DIPUDeviceActivity::instance();

}  // namespace libkineto
