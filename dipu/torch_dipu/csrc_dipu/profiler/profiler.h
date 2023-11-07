#pragma once

#include <csrc_dipu/base/basedef.h>
#include <csrc_dipu/runtime/rthelper.h>

#include <stdint.h>
#include <string>
#include <memory>
#include <chrono>
#include <list>
#include <mutex>
#include <thread>
#include <map>
#include <unordered_map>
#include <deque>
#include <utility>
#include <vector>

#include <IActivityProfiler.h>

#include "CorrelationIDManager.h"

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
    size_t opSeqId;
    string_t attrs;

    ExtraRecordInfo() : scope(""), opSeqId(0), attrs("") {}

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

private:
    RecordsImpl() = default;

public:
    ~RecordsImpl() = default;

    static RecordsImpl& get();
    void addRecord(const Record& record);
    void recordStream(int device, int streamId, const std::string& postfix = "");
    void abandon();

    records_t getAllRecordList() const;
    std::map<std::pair<int64_t, int64_t>, libkineto::ResourceInfo> getResourceInfo() const;
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
    explicit RecordCreator(const string_t& name, size_t opId, uint64_t linkCorrelationId,
                           const ExtraRecordInfo& extraInfo = ExtraRecordInfo());

    ~RecordCreator();

private:
    void end();
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
    DeviceRecordCreator(string_t name, deviceStream_t stream, int streamId, size_t opId, uint64_t linkCorrelationId,
                        const ExtraRecordInfo& extraInfo = ExtraRecordInfo());

    ~DeviceRecordCreator();

private:
    void end();
};

class RecordBlockCreator {
public:
    explicit RecordBlockCreator(string_t name, const ExtraRecordInfo& extraInfo = ExtraRecordInfo(),
                                deviceStream_t stream = dipu::getCurrentDIPUStream(),
                                int streamId = dipu::getCurrentDIPUStream().id(), bool enProfile = isEnable());
    
    void end();

    ~RecordBlockCreator();

private:
    std::unique_ptr<RecordCreator> pHostRecord_ = nullptr;
    std::unique_ptr<DeviceRecordCreator> pDeviceRecord_ = nullptr;
    bool finish_ = false;
};

}  // namespace profile

}  // namespace dipu