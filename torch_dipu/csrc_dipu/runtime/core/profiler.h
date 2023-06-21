#pragma once

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/deviceapis.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/runtime/core/DIPUEvent.h>

#include <string>
#include <memory>
#include <chrono>
#include <list>
#include <utility>

namespace dipu {

namespace profile {
using string_t = std::string;

using clock_t = std::chrono::high_resolution_clock;
using time_point = clock_t::time_point;
typedef std::pair<string_t, size_t> scope_pair_t;

// ----------------------
//
// general functions
//
// ---------------------

/*
 * get the global option
 */
bool isEnable();

/*
 * get profile string
 */
string_t getProfileString();



void FlushAllRecords();
void abandonAllRecords();
size_t timestamp(const time_point& t);


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
    size_t begin;
    size_t end;
    size_t threadIdx;
    ExtraRecordInfo extraInfo;
};


void setThreadName(const string_t& name);
void addRecord(const Record& record);


class RecordCreator final {
private:
    string_t name_;
    size_t opId_;
    time_point begin_;
    bool end_ = true;
    ExtraRecordInfo extraInfo_;

public:
    explicit RecordCreator(const string_t& name, size_t opId = 0,
                           const ExtraRecordInfo& extraInfo = ExtraRecordInfo());

    ~RecordCreator();

private:
    void end();
};

class DeviceEvent;

struct DeviceRecord {
    std::shared_ptr<DeviceEvent> start, stop;
    size_t streamId;
    string_t name;
    size_t opId;
    ExtraRecordInfo extraInfo;
};

class DeviceRecordCreator final {
private:
    string_t name_;
    size_t opId_;
    deviceStream_t stream_;
    std::shared_ptr<DeviceEvent> pStart_, pStop_;
    bool end_ = true;
    ExtraRecordInfo extraInfo_;

public:
    DeviceRecordCreator(string_t name, deviceStream_t stream, size_t opId = 0,
                        const ExtraRecordInfo& extraInfo = ExtraRecordInfo());

    ~DeviceRecordCreator();

private:
    void end();
};


void setProfileOpen(bool profileFlag);


size_t generateId();

void resetId();

class RecordBlockCreator {
public:
    explicit RecordBlockCreator(string_t name, size_t opId = generateId(),
                       const ExtraRecordInfo& extraInfo = ExtraRecordInfo(),
                       deviceStream_t stream = dipu::getCurrentDIPUStream(), bool enProfile = isEnable());
    
    void end();

    ~RecordBlockCreator();

private:
    std::unique_ptr<RecordCreator> pHostRecord_ = nullptr;
    std::unique_ptr<DeviceRecordCreator> pDeviceRecord_ = nullptr;

};

std::list<Record> getRecordList();
void startProfile();
void endProfile();

}  // namespace profile

}  // namespace dipu