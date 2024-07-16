// Copyright (c) 2024, DeepLink.
#pragma once

#include <DeviceActivityInterface.h>
#include <GenericTraceActivity.h>
#include <IActivityProfiler.h>
#include <acl/acl.h>
#include <acl/acl_prof.h>
#include <array>
#include <vector>

#include "basecommimpl.hpp"

namespace dipu {

class AscendDeviceActivity : public libkineto::DeviceActivityInterface {
 public:
  ~AscendDeviceActivity() override = default;
  AscendDeviceActivity(const AscendDeviceActivity&) = delete;
  AscendDeviceActivity& operator=(const AscendDeviceActivity&) = delete;

  // AscendDeviceActivity designed as a singleton
  static AscendDeviceActivity& instance();

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
  AscendDeviceActivity();
  bool remove_temp_dump_path_(const std::string& path);
  aclprofConfig* config_ = nullptr;
  bool enable_ = false;
  std::string current_dump_path_;
  std::string last_dump_path_;
};

}  // namespace dipu
