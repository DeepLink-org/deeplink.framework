#pragma once

#include <deque>
#include <cstdint>

#include "DeviceActivityInterface.h"

namespace dipu {
namespace profile {

class CorrelationIDManager {
 public:
  CorrelationIDManager(const CorrelationIDManager&) = delete;
  CorrelationIDManager& operator=(const CorrelationIDManager&) = delete;

  // CorrelationIDManager designed as a singleton
  static CorrelationIDManager& instance();

  static void pushCorrelationID(
      uint64_t id,
      libkineto::DeviceActivityInterface::CorrelationFlowType type);
  static void popCorrelationID(
      libkineto::DeviceActivityInterface::CorrelationFlowType type);
  static uint64_t getCorrelationID();

 private:
  CorrelationIDManager() = default;

  thread_local static std::array<
      std::deque<uint64_t>, libkineto::DeviceActivityInterface::CorrelationFlowType::End>
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
      external_ids_;
      
  thread_local static std::deque<
      libkineto::DeviceActivityInterface::CorrelationFlowType>
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)    
      type_;
};

}  // namespace profile
}  // namespace dipu
