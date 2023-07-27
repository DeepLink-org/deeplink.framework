#include <iostream>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <deque>
#include <vector>

#include <c10/util/Exception.h>

#include "profiler.h"


namespace dipu {

namespace profile {

static const int32_t DEFAULT_FLUSH_READY_INTERVAL = 1000;

#define STREAM_THREAD_NAME ":Dipu stream "

class DeviceEvent final {
private:
    deviceEvent_t evt_;

public:
    DeviceEvent() {
        dipu::devproxy::createEvent(&evt_);
    }

    ~DeviceEvent() {
        dipu::devproxy::destroyEvent(evt_);
    }

    deviceEvent_t get() const {
        return evt_;
    }

    DeviceEvent(const DeviceEvent&) = delete;
    DeviceEvent& operator=(const DeviceEvent&) = delete;
    DeviceEvent(DeviceEvent&&) = default;
    DeviceEvent& operator=(DeviceEvent&&) = default;
};


namespace {


bool gEnableFlag = false;
bool gIsNano = true;




time_point gBegin = clock_t::now();



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
        beginOffset_ = timestamp(clock_t::now());
    }

    ~StreamTimeOffsetTracker() = default;

    void sync() {
        DeviceEvent end;
        float time;
        dipu::devproxy::recordEvent(end.get(), stream_);
        dipu::devproxy::waitEvent(end.get());
        dipu::devproxy::eventElapsedTime(&time, begin_.get(), end.get());
        size_t endOffset = timestamp(clock_t::now());
        ratio_ = 1.0f * (endOffset - beginOffset_) / time;
    }

    const DeviceEvent& begin() const {
        return begin_;
    }

    size_t offset() const {
        return beginOffset_;
    }

    float ratio() const {
        return ratio_;
    }
};

class RecordsImpl final {
private:
    using records_t = std::list<Record>;
    using mutex_t = std::mutex;

    static mutex_t mut_;
    std::vector<std::unique_ptr<records_t>> allRecordLists_ { 5 };
    std::unordered_map<size_t, string_t> threadName_;
    size_t newIdxs { 0 };

    thread_local static size_t threadIdx;
    thread_local static records_t* pRecords;
    thread_local static bool setup;

    static std::unique_ptr<RecordsImpl> pInstance;
    static const size_t UNDEFINED_THREAD_ID = -1u;

private:
    RecordsImpl() {}

    void reset(const std::lock_guard<std::mutex>& mut_) {
        for (size_t i = 0; i < allRecordLists_.size(); ++i) {
            if (allRecordLists_[i] == nullptr) continue;
            auto& recList = *allRecordLists_[i];
            recList.clear();
        }
    }

public:
    static RecordsImpl& get() {
        if (pInstance == nullptr) {
            std::lock_guard<mutex_t> lk(mut_);
            if (pInstance == nullptr) {
                pInstance.reset(new RecordsImpl());
            }
        }
        return *pInstance;
    }

    static void abandon() {
        std::lock_guard<mutex_t> lk(mut_);
        if (pInstance != nullptr)
            pInstance.release();
    }

    size_t getLocalIdx() {
        checkIdx_();
        return threadIdx;
    }

    records_t& getLocalList_() {
        checkIdx_();
        return *pRecords;
    }

    records_t getAllRecordList_() {
        records_t allrecords;
        for (size_t i = 0; i < allRecordLists_.size(); ++i) {
            if (nullptr != allRecordLists_[i]) {
                records_t* threadRecordList = allRecordLists_[i].get();
                for (auto r : *threadRecordList) {
                    allrecords.push_back(r);
                }
            }
        }
        return allrecords;
    }

private:
    void checkIdx_() {
        if (!setup) {
            std::lock_guard<mutex_t> lk(mut_);
            threadIdx = newIdxs++;
            if (allRecordLists_.size() <= threadIdx) {
                allRecordLists_.resize(2 * threadIdx + 1);
            }
            allRecordLists_[threadIdx].reset(new records_t());
            // set default name to `tid:CPU`
            // can be overwrite by setThreadName
            threadName_[threadIdx] = std::to_string(threadIdx) + ":CPU";
            pRecords = allRecordLists_[threadIdx].get();
            setup = true;
        }
    }


public:
    ~RecordsImpl() {}

public:
    void addRecord(Record record) {
        // Alter threadIdx before insert new Record.
        checkIdx_();
        if (record.threadIdx == UNDEFINED_THREAD_ID) {
            record.threadIdx = threadIdx;
        }
        getLocalList_().push_back(record);
    }

    void setThreadName(const string_t& name, size_t idx = UNDEFINED_THREAD_ID) {
        if (idx == UNDEFINED_THREAD_ID) {
            idx = getLocalIdx();
        }
        std::lock_guard<mutex_t> lk(mut_);
        threadName_[idx] = name;
    }

    bool empty() {
        std::lock_guard<mutex_t> lk(mut_);
        if (allRecordLists_.empty()) {
            return true;
        } else {
            bool isEmpty = true;
            for (size_t i = 0; i < allRecordLists_.size(); i++) {
                if (allRecordLists_[i] == nullptr) continue;
                auto& rec = *allRecordLists_[i];
                isEmpty &= rec.empty();
            }
            return isEmpty;
        }
    }
};

thread_local size_t RecordsImpl::threadIdx = 0;
thread_local RecordsImpl::records_t* RecordsImpl::pRecords = nullptr;
thread_local bool RecordsImpl::setup = false;

std::unique_ptr<RecordsImpl> RecordsImpl::pInstance;
RecordsImpl::mutex_t RecordsImpl::mut_;



class DeviceRecordsImpl final {
private:
    // mutex for records and tracker
    static std::mutex mtx_;
    std::list<DeviceRecord> records_;
    std::vector<Record> ready_records_;
    std::unique_ptr<StreamTimeOffsetTracker> pTracker_;

    std::unordered_map<size_t, size_t> streamName_;

    static std::unique_ptr<DeviceRecordsImpl> pInstance_;

private:
    DeviceRecordsImpl() {}

    static bool enableFlushReady() {
        static bool enable_flush_ready = (std::getenv("DIPU_FLUSH_READY") != nullptr);
        return enable_flush_ready;
    }

    static int32_t flushReadyInterval() {
        static int32_t flush_ready_interval = []() -> int32_t {
            const char* str = std::getenv("DIPU_FLUSH_READY_INTERVAL");
            if (str == nullptr) {
                return DEFAULT_FLUSH_READY_INTERVAL;
            }
            return std::stoi(str);
        }();
        return flush_ready_interval;
    }

    deviceEvent_t beginEvent() const {
        TORCH_CHECK(pTracker_, "dipu profiler error with pTracker is not inited");
        return pTracker_->begin().get();
    }

    size_t getTime(const DeviceEvent& evt,
                   float scale = 1., size_t shift = 0) {
        float time;
        dipu::devproxy::waitEvent(evt.get());
        dipu::devproxy::eventElapsedTime(&time, beginEvent(), evt.get());
        return static_cast<size_t>(time * scale) + shift;
    }

    void ensureStreamName(size_t streamId) {
        if (streamName_.find(streamId) == streamName_.end()) {
            size_t count = streamName_.size();
            size_t idx = count + 90090000u;
            streamName_[streamId] = idx;
            auto& impl = RecordsImpl::get();
            impl.setThreadName(
                    std::to_string(impl.getLocalIdx()) + STREAM_THREAD_NAME + std::to_string(count), idx);
        }
    }

public:
    ~DeviceRecordsImpl() {
        flush();
    }

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
        if (enableFlushReady() && (records_.size() % flushReadyInterval() == 0)) {
            flushReady();
        }
    }

    void flushReady() {
        while (records_.size() > 0) {
            auto& r = records_.front();
            ensureStreamName(r.streamId);
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
            ready_records_.push_back(Record({r.name, r.opId,
                                         static_cast<size_t>(t1 * 1e3),
                                         static_cast<size_t>((t1 + t2) * 1e3),
                                         r.streamId,
                                         r.extraInfo}));
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
                r.threadIdx = streamName_[r.threadIdx];
                addRecord(r);
            }
            ready_records_.clear();

            for (auto& r : records_) {
                ensureStreamName(r.streamId);
                addRecord(Record({r.name, r.opId,
                                  getTime(*r.start, ratio, offset),
                                  getTime(*r.stop, ratio, offset),
                                  streamName_[r.streamId], r.extraInfo}));
            }
            reset();
        }
    }

    void reset() {
        records_.clear();
        pTracker_.reset();
    }

    static void abandon() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (pInstance_ != nullptr)
            pInstance_.release();
    }

    static DeviceRecordsImpl& get() {
        if (pInstance_ == nullptr) {
            std::lock_guard<std::mutex> lk(mtx_);
            if (pInstance_ == nullptr)
                pInstance_.reset(new DeviceRecordsImpl());
        }
        return *pInstance_;
    }
};


std::unique_ptr<DeviceRecordsImpl> DeviceRecordsImpl::pInstance_;
std::mutex DeviceRecordsImpl::mtx_;



}  // end namespace


bool isEnable() {
    return gEnableFlag;
}

void FlushAllRecords() {
    DeviceRecordsImpl::get().flush();
}

void abandonAllRecords() {
    RecordsImpl::abandon();
    DeviceRecordsImpl::abandon();

}

void setProfileOpen(bool profileFlag) {
    gEnableFlag = profileFlag;
}


thread_local std::string sProfileScopeName = "";
thread_local size_t sProfileScopeId = 0;
thread_local size_t moduleId = 10000;  // avoid clash with pytorch id

void setScopePair(const std::string& name, size_t id) {
    sProfileScopeName = name;
    sProfileScopeId = id;
}

size_t generateId() {
    return ++moduleId;
}

void resetId() {
    moduleId = 10000;
}

size_t timestamp(const time_point& t) {
    if (gIsNano) {
        using ut = std::chrono::duration<size_t, std::nano>;
        return std::chrono::duration_cast<ut>(t - gBegin).count();
    } else {
        using ut = std::chrono::duration<size_t, std::micro>;
        return std::chrono::duration_cast<ut>(t - gBegin).count();
    }
}


void setThreadName(const string_t& name) {
    RecordsImpl::get().setThreadName(name);
}

void addRecord(const Record& record) {
    RecordsImpl::get().addRecord(record);
}

RecordCreator::RecordCreator(const string_t& name, size_t opId,
                             const ExtraRecordInfo& extraInfo) {
    if (isEnable()) {
        name_ = name;
        opId_ = opId;
        begin_ = clock_t::now();
        end_ = false;
        extraInfo_ = extraInfo;
    }
}

RecordCreator::~RecordCreator() {
    end();
}

void RecordCreator::end() {
    if (!end_) {
        addRecord(Record{name_, opId_, timestamp(begin_),
                            timestamp(clock_t::now()), -1u, extraInfo_});
    }
    end_ = true;
}


DeviceRecordCreator::DeviceRecordCreator(string_t name, deviceStream_t stream, size_t opId,
                                         const ExtraRecordInfo& extraInfo) {
    if (isEnable()) {
        DeviceRecordsImpl::get().ensureSetup(stream);
        name_ = name;
        opId_ = opId;
        extraInfo_ = extraInfo;
        stream_ = stream;
        pStart_.reset(new DeviceEvent());
        pStop_.reset(new DeviceEvent());
        dipu::devproxy::recordEvent(pStart_->get(), stream_);
        end_ = false;
    }
}

DeviceRecordCreator::~DeviceRecordCreator() {
    end();
}

void DeviceRecordCreator::end() {
    if (!end_) {
        TORCH_CHECK(pStart_, "dipu profiler error with pStart_ is not inited");
        TORCH_CHECK(pStop_, "dipu profiler error with pStop_ is not inited");
        dipu::devproxy::recordEvent(pStop_->get(), stream_);
        DeviceRecordsImpl::get().addDeviceRecord(DeviceRecord{
                pStart_, pStop_, (size_t)stream_,
                name_, opId_, extraInfo_});
    }
    end_ = true;
}


RecordBlockCreator::RecordBlockCreator(string_t name, size_t opId,
                                       const ExtraRecordInfo& extraInfo,
                                       deviceStream_t stream, bool enProfile) {
    if (enProfile && isEnable()) {
        pHostRecord_.reset(new RecordCreator(name, opId, extraInfo));
        pDeviceRecord_.reset(new DeviceRecordCreator(name, stream,
                                                     opId, extraInfo));
    }
}

void RecordBlockCreator::end() {
    pHostRecord_.reset();
    pDeviceRecord_.reset();
}

RecordBlockCreator::~RecordBlockCreator() {
    pHostRecord_.reset();
    pDeviceRecord_.reset();
}


std::list<Record> getRecordList() {
    return RecordsImpl::get().getAllRecordList_();
}

void startProfile() {
    gEnableFlag = true;
    gBegin = clock_t::now();
}

void endProfile() {
    abandonAllRecords();
    gEnableFlag = false;
}

}  // namespace profile

}  // namespace dipu
