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

struct ExtraRecordInfo {
  string_t scope;
  size_t opSeqId{};
  string_t attrs;

  ExtraRecordInfo& setScope(const string_t& scopeName) {
    scope = scopeName;
    return *this;
  }

  ExtraRecordInfo& setSeqId(size_t seqId) {
    opSeqId = seqId;
    return *this;
  }

  ExtraRecordInfo& setAttrs(const string_t& sAttrs) {
    attrs = sAttrs;
    return *this;
  }
};

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
  ExtraRecordInfo extraInfo;
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
  ExtraRecordInfo extraInfo_;

 public:
  explicit RecordCreator(string_t name, size_t opId, uint64_t linkCorrelationId,
                         ExtraRecordInfo extraInfo = ExtraRecordInfo());

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
  ExtraRecordInfo extraInfo;
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
  ExtraRecordInfo extraInfo_;

 public:
  DeviceRecordCreator(string_t name, deviceStream_t stream, int streamId,
                      size_t opId, uint64_t linkCorrelationId,
                      ExtraRecordInfo extraInfo = ExtraRecordInfo());

  ~DeviceRecordCreator() { end(); }

 private:
  void end() noexcept;
};

class RecordBlockCreator {
 public:
  // TODO(lljbash): maybe use std::string_view and std::optional after c++17
  explicit RecordBlockCreator(
      c10::string_view name,
      c10::optional<ExtraRecordInfo> extraInfo = c10::nullopt,
      c10::optional<deviceStream_t> stream = c10::nullopt,
      c10::optional<c10::StreamId> streamId = c10::nullopt,
      c10::optional<bool> enProfile = c10::nullopt) {
    if (enProfile.value_or(isEnable())) {
      if (!extraInfo) {
        extraInfo.emplace();
      }
      if (!stream) {
        auto dipu_stream = getCurrentDIPUStream();
        if (!streamId) {
          streamId = dipu_stream.id();
        }
        stream = dipu_stream.rawstream();
      }
      initialize(string_t(name), std::move(*extraInfo), *stream, *streamId);
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
  void initialize(string_t name, ExtraRecordInfo extraInfo,
                  deviceStream_t stream, c10::StreamId streamId);

  std::unique_ptr<RecordCreator> pHostRecord_ = nullptr;
  std::unique_ptr<DeviceRecordCreator> pDeviceRecord_ = nullptr;
  bool finish_ = false;
};

}  // namespace profile

}  // namespace dipu
