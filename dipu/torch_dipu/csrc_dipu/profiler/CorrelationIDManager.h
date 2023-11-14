#pragma once

#include <DeviceActivityInterface.h>
#include <deque>
#include <stdint.h>

namespace dipu {
namespace profile {

class CorrelationIDManager {
 public:
  CorrelationIDManager(const CorrelationIDManager &) = delete;
  CorrelationIDManager &operator=(const CorrelationIDManager &) = delete;

  // CorrelationIDManager designed as a singleton
  static CorrelationIDManager &instance();

  void pushCorrelationID(
      uint64_t id,
      libkineto::DeviceActivityInterface::CorrelationFlowType type);
  void popCorrelationID(
      libkineto::DeviceActivityInterface::CorrelationFlowType type);
  uint64_t getCorrelationID() const;

 private:
  CorrelationIDManager() = default;

 private:
  thread_local static std::deque<uint64_t> external_ids_
      [libkineto::DeviceActivityInterface::CorrelationFlowType::End];
  thread_local static std::deque<
      libkineto::DeviceActivityInterface::CorrelationFlowType>
      type_;
};

}  // namespace profile
}  // namespace dipu