// Copyright (c) 2024, DeepLink.
#include "AscendDeviceActivity.h"

#include <ctime>
#include <dirent.h>
#include <output_base.h>
#include <sys/stat.h>
#include <unistd.h>

#include "csrc_dipu/base/environ.hpp"
#include "csrc_dipu/utils/Log.h"

namespace dipu {

static const uint64_t kNpuEvents = 431;
static const uint64_t kAicoreMetrics = 1;

using libkineto::GenericTraceActivity;

AscendDeviceActivity::AscendDeviceActivity() {}

AscendDeviceActivity& AscendDeviceActivity::instance() {
  static AscendDeviceActivity instance;
  return instance;
}

// override pure virtual function, do nothing
void AscendDeviceActivity::pushCorrelationID(
    uint64_t id, DeviceActivityInterface::CorrelationFlowType type) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::popCorrelationID(
    DeviceActivityInterface::CorrelationFlowType type) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::enableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::disableActivities(
    const std::set<libkineto::ActivityType>& selected_activities) {}

// override pure virtual function, do nothing
void AscendDeviceActivity::clearActivities() {}

bool AscendDeviceActivity::remove_temp_dump_path_(const std::string& path) {
  std::unique_ptr<DIR, decltype(&closedir)> dir(opendir(path.c_str()),
                                                &closedir);
  if (!dir) {
    return false;
  }

  dirent* entry;
  while ((entry = readdir(dir.get())) != nullptr) {
    std::string entry_name = entry->d_name;
    if (entry_name == "." || entry_name == "..") {
      continue;
    }

    std::string entry_path = path + "/" + entry_name;
    struct stat entry_stat;
    if (stat(entry_path.c_str(), &entry_stat) == -1) {
      return false;
    }

    if (S_ISDIR(entry_stat.st_mode)) {
      if (!remove_temp_dump_path_(entry_path)) {
        return false;
      }
    } else {
      if (remove(entry_path.c_str()) != 0) {
        return false;
      }
    }
  }
  return rmdir(path.c_str()) == 0;
}

int32_t AscendDeviceActivity::processActivities(
    libkineto::ActivityLogger& logger,
    std::function<const libkineto::ITraceActivity*(int32_t)> linked_activity,
    int64_t start_time, int64_t end_time) {
  struct timespec tx;
  struct timespec ts;
  struct timespec ty;

  DIPU_CALLACLRT(aclprofFinalize());

  // clang-format off
  // the difference between torch realtime and monotonic_raw,
  // as well as the difference between aclprof realtime and monotonic_raw,
  // are statistically calculated to align the timestamps of aclprof and torch
  // for example: torch's time difference is 10, and aclprof's time difference
  // is 3, we assume that monotonic_raw time is the correct baseline, then,
  // the time diff between torch and aclprof is 10 - 3 = 7 
  // aclprof's time difference is in generated file by msprof tool: PROF_XXXX/host/end_info
  // clang-format on

  clock_gettime(CLOCK_REALTIME, &tx);
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  clock_gettime(CLOCK_REALTIME, &ty);

  // clang-format off
  // time interval between these two calculation(aclprof and torch) is the smaller the better
  // and aclprof will generate its time difference in start_info and end_info
  // the reason why I use aclprof end_info and put the calculation right after aclprofFinalize is that,
  // after test, this is more precise and less stable than calculate at startTrace() and compare with start_info
  // clang-format on

  const int S_TO_NS_FACTOR = 1000000000;
  auto time_x = tx.tv_sec * S_TO_NS_FACTOR + tx.tv_nsec;
  auto monotonic_raw = ts.tv_sec * S_TO_NS_FACTOR + ts.tv_nsec;
  auto time_y = ty.tv_sec * S_TO_NS_FACTOR + ty.tv_nsec;

  int64_t diff = (time_x >> 1) + (time_y >> 1) - monotonic_raw;

  GenericTraceActivity time_diff;
  time_diff.activityName = "torch_time_diff:" + std::to_string(diff);
  time_diff.activityType = libkineto::ActivityType::USER_ANNOTATION;
  time_diff.startTime = start_time;
  time_diff.endTime = end_time;

  GenericTraceActivity tmp_path;
  tmp_path.activityName = "random_temp_dir:" + current_dump_path_;
  tmp_path.activityType = libkineto::ActivityType::USER_ANNOTATION;
  tmp_path.startTime = start_time;
  tmp_path.endTime = end_time;
  logger.handleGenericActivity(tmp_path);
  logger.handleGenericActivity(time_diff);

  std::string temp_path_prefix = "./tmp/aclprof";
  if (last_dump_path_.compare(0, temp_path_prefix.size(), temp_path_prefix) ==
      0) {
    if (remove_temp_dump_path_(last_dump_path_) == false) {
      DIPU_LOGW(
          "remove ascend profiler temp file failed, may need to remove "
          "manually");
    }
  }

  return 0;
}

void AscendDeviceActivity::startTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  if (enable_) {
    DIPU_LOGW("ascend profiler has already enabled");
    return;
  }
  enable_ = true;

  last_dump_path_ = current_dump_path_;

  struct stat st;
  if (stat("./tmp", &st) == -1) {
    mkdir("./tmp", 0644);
  }
  if (stat("./tmp/aclprof", &st) == -1) {
    mkdir("./tmp/aclprof", 0644);
  }

  char dump_path_template[] = "./tmp/aclprof/aclprofXXXXXX";
  char* dump_path_cstring = mkdtemp(dump_path_template);

  if (dump_path_cstring != nullptr) {
    current_dump_path_ = dump_path_cstring;
  } else {
    DIPU_LOGE(
        "aclprof random dump path generate failed, the export results may be "
        "incorrect");
    current_dump_path_ = "./tmp/aclprof/aclprof_error";
  }

  DIPU_CALLACLRT(
      aclprofInit(current_dump_path_.c_str(), current_dump_path_.size()));
  DIPU_CALLACLRT(aclrtSynchronizeDevice());

  int32_t device_index = 0;
  DIPU_CALLACLRT(aclrtGetDevice(&device_index));

  std::array<uint32_t, 1> device_ids = {static_cast<uint32_t>(device_index)};
  aclprofAicoreEvents* events = nullptr;
  config_ = aclprofCreateConfig(
      device_ids.data(), device_ids.size(),
      static_cast<aclprofAicoreMetrics>(kAicoreMetrics), events, kNpuEvents);
  TORCH_CHECK(config_ != nullptr,
              "aclprofCreateConfig fail, device_index = ", device_index,
              "npu_event = ", kNpuEvents, "aicore_metrics = ", kAicoreMetrics);

  DIPU_CALLACLRT(aclprofStart(config_));
}

void AscendDeviceActivity::stopTrace(
    const std::set<libkineto::ActivityType>& selected_activities) {
  if (!enable_) {
    DIPU_LOGW("ascend profiler has already disabled");
    return;
  }

  DIPU_CALLACLRT(aclrtSynchronizeDevice());
  DIPU_CALLACLRT(aclprofStop(config_));
  DIPU_CALLACLRT(aclprofDestroyConfig(config_))

  config_ = nullptr;
  enable_ = false;
}

// override pure virtual function, do nothing
void AscendDeviceActivity::teardownContext() {}

// override pure virtual function, do nothing
void AscendDeviceActivity::setMaxBufferSize(int32_t size) {}

// NOLINTNEXTLINE(cppcoreguidelines-interfaces-global-init)
const static int32_t Ascend_device_activity_init = []() {
  if (!dipu::environ::detail::getEnvOrDefault<bool>("FORCE_USE_DIPU_PROFILER",
                                                    false)) {
    libkineto::device_activity_singleton = &AscendDeviceActivity::instance();
    return 1;
  }
  return 0;
}();

}  // namespace dipu
