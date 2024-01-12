#include "CorrelationIDManager.h"

#include <DeviceActivityInterface.h>

namespace dipu {
namespace profile {

using libkineto::DeviceActivityInterface;

CorrelationIDManager& CorrelationIDManager::instance() {
  static CorrelationIDManager instance;
  return instance;
}

void CorrelationIDManager::pushCorrelationID(
    uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {
  external_ids_[type].emplace_back(id);
  type_.push_back(type);
}

void CorrelationIDManager::popCorrelationID(
    DeviceActivityInterface::CorrelationFlowType type) {
  external_ids_[type].pop_back();
  type_.pop_back();
}

uint64_t CorrelationIDManager::getCorrelationID() {
  DeviceActivityInterface::CorrelationFlowType type = type_.back();
  return external_ids_[type].back();
}

thread_local std::array<std::deque<uint64_t>,
                        DeviceActivityInterface::CorrelationFlowType::End>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    CorrelationIDManager::external_ids_;

thread_local std::deque<DeviceActivityInterface::CorrelationFlowType>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    CorrelationIDManager::type_;

}  // namespace profile
}  // namespace dipu
