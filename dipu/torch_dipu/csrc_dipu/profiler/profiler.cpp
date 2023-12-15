#include "profiler.h"

#include <cstdio>
#include <memory>
#include <utility>

#include <c10/util/Exception.h>
#include <c10/util/string_view.h>
#include <torch/csrc/profiler/util.h>

#include "csrc_dipu/profiler/CorrelationIDManager.h"

#include "ThreadUtil.h"

namespace dipu {

namespace profile {

static const int32_t DEFAULT_FLUSH_READY_INTERVAL = 1000;

class DeviceEvent final {
 private:
  deviceEvent_t evt_;

 public:
  DeviceEvent() { dipu::devproxy::createEvent(&evt_); }

  ~DeviceEvent() { dipu::devproxy::destroyEvent(evt_); }

  deviceEvent_t get() const { return evt_; }

  DeviceEvent(const DeviceEvent&) = delete;
  DeviceEvent& operator=(const DeviceEvent&) = delete;
  DeviceEvent(DeviceEvent&&) = default;
  DeviceEvent& operator=(DeviceEvent&&) = default;
};

class StreamTimeOffsetTracker final {
  DeviceEvent begin_;
  deviceStream_t stream_;
  size_t beginOffset_;
  float ratio_ = 0.f;

 public:
  explicit StreamTimeOffsetTracker(deviceStream_t stream) {
    stream_ = stream;
    devproxy::recordEvent(begin_.get(), stream_);
    devproxy::waitEvent(begin_.get());
    beginOffset_ = torch::profiler::impl::getTime();
  }

  ~StreamTimeOffsetTracker() = default;

  void sync() {
    DeviceEvent end;
    float time;
    dipu::devproxy::recordEvent(end.get(), stream_);
    dipu::devproxy::waitEvent(end.get());
    dipu::devproxy::eventElapsedTime(&time, begin_.get(), end.get());
    size_t endOffset = torch::profiler::impl::getTime();
    ratio_ = 1.0f * (endOffset - beginOffset_) / time;
  }

  const DeviceEvent& begin() const { return begin_; }

  size_t offset() const { return beginOffset_; }

  float ratio() const { return ratio_; }
};

RecordsImpl& RecordsImpl::get() {
  static RecordsImpl instance;
  return instance;
}

void RecordsImpl::abandon() {
  std::lock_guard<mutex_t> lck(mtx_);
  for (auto& kv : allRecordLists_) {
    kv.second->clear();
  }
  resourceInfo_.clear();
}

void RecordsImpl::addRecord(const Record& record) {
  if (pRecords == nullptr) {
    std::lock_guard<mutex_t> lk(mtx_);
    int32_t tid = libkineto::systemThreadId();
    allRecordLists_[tid] = std::make_unique<records_t>();
    pRecords = allRecordLists_[tid].get();
  }
  pRecords->emplace_back(record);
}

void RecordsImpl::recordStream(int device, int streamId,
                               const std::string& postfix) {
  std::lock_guard<mutex_t> lck(mtx_);
  if (resourceInfo_.find({device, streamId}) == resourceInfo_.end()) {
    resourceInfo_.emplace(std::make_pair(device, streamId),
                          libkineto::ResourceInfo(
                              device, streamId, streamId,
                              fmt::format("stream {} {}", streamId, postfix)));
  }
}

RecordsImpl::records_t RecordsImpl::getAllRecordList() const {
  std::lock_guard<mutex_t> lck(mtx_);
  records_t allrecords;
  for (const auto& kv : allRecordLists_) {
    if (!kv.second || kv.second->empty()) {
      continue;
    }

    for (const auto& r : *(kv.second)) {
      allrecords.push_back(r);
    }
  }
  return allrecords;
}

std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo>
RecordsImpl::getResourceInfo() const {
  std::lock_guard<mutex_t> lck(mtx_);
  return resourceInfo_;
}

thread_local RecordsImpl::records_t* RecordsImpl::pRecords = nullptr;

class DeviceRecordsImpl final {
 private:
  // mutex for records and tracker
  std::mutex mtx_;
  std::list<DeviceRecord> records_;
  std::vector<Record> ready_records_;
  std::unique_ptr<StreamTimeOffsetTracker> pTracker_;

 private:
  DeviceRecordsImpl() {}

  static bool enableFlushReadyEvent() {
    static bool enable_flush_ready =
        (std::getenv("DIPU_DISABLE_FLUSH_READY_EVENT") == nullptr);
    return enable_flush_ready;
  }

  static int32_t flushReadyEventInterval() {
    static int32_t flush_ready_event_interval = []() -> int32_t {
      const char* str = std::getenv("DIPU_FLUSH_READY_EVENT_INTERVAL");
      return str == nullptr ? DEFAULT_FLUSH_READY_INTERVAL : std::stoi(str);
    }();
    return flush_ready_event_interval;
  }

  deviceEvent_t beginEvent() const {
    TORCH_CHECK(pTracker_, "dipu profiler error with pTracker is not inited");
    return pTracker_->begin().get();
  }

  size_t getTime(const DeviceEvent& evt, float scale = 1., size_t shift = 0) {
    float time;
    dipu::devproxy::waitEvent(evt.get());
    dipu::devproxy::eventElapsedTime(&time, beginEvent(), evt.get());
    return static_cast<size_t>(time * scale) + shift;
  }

 public:
  ~DeviceRecordsImpl() { reset(); }

 public:
  void ensureSetup(deviceStream_t stream) {
    if (!pTracker_) {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!pTracker_) {
        pTracker_.reset(new StreamTimeOffsetTracker(stream));
      }
    }
  }

  void addDeviceRecord(DeviceRecord record) {
    std::lock_guard<std::mutex> lk(mtx_);
    TORCH_CHECK(pTracker_, "dipu profiler error with pTracker is not inited");
    records_.push_back(record);
    if (enableFlushReadyEvent() &&
        (records_.size() % flushReadyEventInterval() == 0)) {
      flushReady();
    }
  }

  void flushReady() {
    while (records_.size() > 0) {
      auto& r = records_.front();
      auto start_status = dipu::devproxy::getEventStatus(r.start->get());
      auto end_status = dipu::devproxy::getEventStatus(r.stop->get());
      auto origin_status = dipu::devproxy::getEventStatus(beginEvent());
      if (start_status != devapis::EventStatus::READY ||
          end_status != devapis::EventStatus::READY ||
          origin_status != devapis::EventStatus::READY) {
        break;
      }
      float t1 = 0.0f;
      float t2 = 0.0f;
      dipu::devproxy::eventElapsedTime(&t1, beginEvent(), r.start->get());
      dipu::devproxy::eventElapsedTime(&t2, r.start->get(), r.stop->get());
      ready_records_.push_back(
          Record({r.name, r.opId, static_cast<size_t>(t1 * 1e3),
                  static_cast<size_t>((t1 + t2) * 1e3), r.deviceId, r.streamId,
                  true, r.linkCorrelationId, r.extraInfo}));
      records_.pop_front();
    }
  }

  void flush() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (records_.size() > 0) {
      TORCH_CHECK(pTracker_, "dipu profiler error with pTracker is not inited");
      auto& trakcer = *pTracker_;
      trakcer.sync();
      float ratio = trakcer.ratio();
      size_t offset = trakcer.offset();

      for (auto& r : ready_records_) {
        r.begin = static_cast<size_t>(r.begin * 1e-3 * ratio) + offset;
        r.end = static_cast<size_t>(r.end * 1e-3 * ratio) + offset;
        RecordsImpl::get().addRecord(r);
      }
      ready_records_.clear();

      for (auto& r : records_) {
        RecordsImpl::get().addRecord(
            Record({r.name, r.opId, getTime(*r.start, ratio, offset),
                    getTime(*r.stop, ratio, offset), r.deviceId, r.streamId,
                    true, r.linkCorrelationId, r.extraInfo}));
      }
      records_.clear();
    }
  }

  void reset() {
    std::lock_guard<std::mutex> lck(mtx_);
    records_.clear();
    ready_records_.clear();
    pTracker_.reset();
  }

  void abandon() { reset(); }

  static DeviceRecordsImpl& get() {
    static DeviceRecordsImpl instance;
    return instance;
  }
};

bool gEnableFlag = false;

bool isEnable() { return gEnableFlag; }

void setProfileOpen(bool profileFlag) { gEnableFlag = profileFlag; }

void FlushAllRecords() { DeviceRecordsImpl::get().flush(); }

static size_t kInitModuleId = 10000;
std::atomic<size_t> moduleId(kInitModuleId);

size_t generateId() { return ++moduleId; }

void resetId() { moduleId = kInitModuleId; }

void abandonAllRecords() {
  RecordsImpl::get().abandon();
  DeviceRecordsImpl::get().abandon();
  resetId();
}

RecordCreator::RecordCreator(string_t name, size_t opId,
                             uint64_t linkCorrelationId,
                             ExtraRecordInfo extraInfo) {
  if (isEnable()) {
    name_ = std::move(name);
    opId_ = opId;
    begin_ = torch::profiler::impl::getTime();
    end_ = false;
    linkCorrelationId_ = linkCorrelationId;
    extraInfo_ = std::move(extraInfo);
  }
}

void RecordCreator::end() noexcept {
  if (!end_) {
    RecordsImpl::get().addRecord(
        Record{name_, opId_, begin_,
               static_cast<size_t>(torch::profiler::impl::getTime()),
               static_cast<size_t>(libkineto::processId()),
               static_cast<size_t>(libkineto::systemThreadId()), false,
               linkCorrelationId_, extraInfo_});
  }
  end_ = true;
}

DeviceRecordCreator::DeviceRecordCreator(string_t name, deviceStream_t stream,
                                         int streamId, size_t opId,
                                         uint64_t linkCorrelationId,
                                         ExtraRecordInfo extraInfo) {
  if (isEnable()) {
    DeviceRecordsImpl::get().ensureSetup(stream);
    name_ = std::move(name);
    opId_ = opId;
    extraInfo_ = std::move(extraInfo);
    stream_ = stream;
    streamId_ = streamId;
    pStart_.reset(new DeviceEvent());
    pStop_.reset(new DeviceEvent());
    dipu::devproxy::recordEvent(pStart_->get(), stream_);
    linkCorrelationId_ = linkCorrelationId;
    end_ = false;
  }
}

void DeviceRecordCreator::end() noexcept {
  if (!end_) {
    TORCH_CHECK(pStart_, "dipu profiler error with pStart_ is not inited");
    TORCH_CHECK(pStop_, "dipu profiler error with pStop_ is not inited");
    dipu::devproxy::recordEvent(pStop_->get(), stream_);
    auto deviceId = dipu::devproxy::current_device();
    DeviceRecordsImpl::get().addDeviceRecord(
        DeviceRecord{pStart_, pStop_, static_cast<size_t>(deviceId),
                     static_cast<size_t>(streamId_), name_, opId_,
                     linkCorrelationId_, extraInfo_});
    RecordsImpl::get().recordStream(deviceId, streamId_);
  }
  end_ = true;
}

static std::string extraceFunction(const std::string& functionName) {
  auto start = functionName.find_first_not_of(':');
  if (start == std::string::npos) {
    return "";
  }

  auto end = functionName.find_first_of('(');
  if (end == std::string::npos) {
    end = functionName.size();
  }

  if (end <= start) {
    return "";
  }
  return functionName.substr(start, end - start);
}

void RecordBlockCreator::initialize(string_t name, ExtraRecordInfo extraInfo,
                                    deviceStream_t stream,
                                    c10::StreamId streamId) {
  size_t opId = generateId();
  uint64_t correlationId = CorrelationIDManager::instance().getCorrelationID();
  name = extraceFunction(name);
  pHostRecord_ = std::make_unique<RecordCreator>("LaunchKernel_" + name, opId,
                                                 correlationId, extraInfo);
  pDeviceRecord_ = std::make_unique<DeviceRecordCreator>(
      std::move(name), stream, streamId, opId, correlationId,
      std::move(extraInfo));
}
}  // namespace profile

}  // namespace dipu
