#pragma once

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include <c10/core/Stream.h>
#include <c10/util/Optional.h>
#include <c10/util/string_view.h>

#include "csrc_dipu/vendor/vendorapi.h"
#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/runtime/core/ObjectPool.h>
#include <csrc_dipu/runtime/rthelper.h>

#include "IActivityProfiler.h"

namespace dipu {
namespace profile {
using string_t = std::string;

// ----------------------
//
// general functions
//
// ---------------------

/*
 * get the global option
 */
bool isEnable();
void setProfileOpen(bool profileFlag);

void FlushAllRecords();
void abandonAllRecords();

struct Record {
  string_t name;
  size_t opId;
  // clock real time in nanosecond
  size_t begin;
  size_t end;
  size_t pid;
  size_t threadIdx;
  bool isKernel = false;
  uint64_t linkCorrelationId = 0;
};

class RecordsImpl final {
 private:
  using records_t = std::list<Record>;
  using mutex_t = std::mutex;

  mutable mutex_t mtx_;
  // tid -> record list
  std::unordered_map<int32_t, std::unique_ptr<records_t>> allRecordLists_;
  thread_local static records_t* pRecords;

  std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo> resourceInfo_;

  RecordsImpl() = default;

 public:
  ~RecordsImpl() = default;

  static RecordsImpl& get();
  void addRecord(const Record& record);
  void recordStream(int device, int streamId, const std::string& postfix = "");
  void abandon();

  records_t getAllRecordList() const;
  std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo>
  getResourceInfo() const;
};

class RecordCreator final {
 private:
  string_t name_;
  size_t opId_;
  size_t begin_;
  bool end_ = true;
  uint64_t linkCorrelationId_ = 0;

 public:
  RecordCreator() = default;
  RecordCreator(string_t name, size_t opId, uint64_t linkCorrelationId);

  ~RecordCreator() { end(); }

 private:
  void end() noexcept;
};

class DeviceEvent;

struct DeviceRecord {
  std::shared_ptr<DeviceEvent> start, stop;
  size_t deviceId;
  size_t streamId;
  string_t name;
  size_t opId;
  uint64_t linkCorrelationId = 0;
};

class DeviceRecordCreator final {
 private:
  string_t name_;
  size_t opId_;
  deviceStream_t stream_;
  int streamId_;
  std::shared_ptr<DeviceEvent> pStart_, pStop_;
  bool end_ = true;
  uint64_t linkCorrelationId_ = 0;

 public:
  DeviceRecordCreator() = default;
  DeviceRecordCreator(string_t name, deviceStream_t stream, int streamId,
                      size_t opId, uint64_t linkCorrelationId);

  ~DeviceRecordCreator() { end(); }

 private:
  void end() noexcept;
};

extern ObjectPool<RecordCreator> record_creator_pool;
extern ObjectPool<DeviceRecordCreator> device_record_creator_pool;

class RecordBlockCreator {
 public:
  RecordBlockCreator() = default;
  // TODO(lljbash): maybe use std::string_view and std::optional after c++17
  explicit RecordBlockCreator(
      c10::string_view name,
      c10::optional<deviceStream_t> stream = c10::nullopt,
      c10::optional<c10::StreamId> streamId = c10::nullopt,
      c10::optional<bool> enProfile = c10::nullopt) {
    if (enProfile.value_or(isEnable())) {
      if (!stream) {
        auto dipu_stream = getCurrentDIPUStream();
        if (!streamId) {
          streamId = dipu_stream.id();
        }
        stream = static_cast<deviceStream_t>(dipu_stream);
      }
      initialize(string_t(name), *stream, *streamId);
    }
  }

  void end() noexcept {
    if (!finish_) {
      pHostRecord_.reset();
      pDeviceRecord_.reset();
      finish_ = true;
    }
  }

  ~RecordBlockCreator() { end(); }

 private:
  void initialize(string_t name, deviceStream_t stream, c10::StreamId streamId);

  struct RecordCreatorDeleter {
    void operator()(RecordCreator* record) const {
      if (record != nullptr) {
        record_creator_pool.free(record);
      }
    }
  };

  struct DeviceRecordCreatorDeleter {
    void operator()(DeviceRecordCreator* record) const {
      if (record != nullptr) {
        device_record_creator_pool.free(record);
      }
    }
  };

  std::unique_ptr<RecordCreator, RecordCreatorDeleter> pHostRecord_ = nullptr;
  std::unique_ptr<DeviceRecordCreator, DeviceRecordCreatorDeleter>
      pDeviceRecord_ = nullptr;
  bool finish_ = false;
};

extern ObjectPool<RecordBlockCreator> record_block_creator_pool;

}  // namespace profile

}  // namespace dipu
