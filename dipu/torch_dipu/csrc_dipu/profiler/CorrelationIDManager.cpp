#include "CorrelationIDManager.h"

namespace dipu {
namespace profile {

using libkineto::DeviceActivityInterface;

CorrelationIDManager& CorrelationIDManager::instance() {
  static CorrelationIDManager instance;
  return instance;
}

void CorrelationIDManager::pushCorrelationID(uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {
    external_ids_[type].emplace_back(id);
    type_.push_back(type);
}

void CorrelationIDManager::popCorrelationID(DeviceActivityInterface::CorrelationFlowType type) {
    external_ids_[type].pop_back();
    type_.pop_back();
}

uint64_t CorrelationIDManager::getCorrelationID() const {
    DeviceActivityInterface::CorrelationFlowType type = type_.back();
    return external_ids_[type].back();
}

thread_local std::deque<uint64_t> CorrelationIDManager::external_ids_[DeviceActivityInterface::CorrelationFlowType::End];
thread_local std::deque<DeviceActivityInterface::CorrelationFlowType> CorrelationIDManager::type_;

}  // namespace profile
}  // namespace dipu