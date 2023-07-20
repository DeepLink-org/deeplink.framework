#include "DIPUEventPool.h"
#include <deque>
#include <mutex>
#include <functional>
#include <iostream>

namespace dipu {

template<typename T>
class EventPool final {
protected:
    std::deque<T> event_pool_;
    unsigned int allocate_num_ = 0;

    std::function<void(T&)> allocator_;
    std::function<void(T&)> deleter_;
    using mutex_t = std::recursive_mutex;
    mutex_t event_mutex_;

public:
    EventPool(const std::function<void(T&)>& allocator,
            const std::function<void(T&)>& deleter)
            : allocator_(allocator), deleter_(deleter) {
    }

    EventPool(const EventPool&) = delete;
    EventPool(EventPool&&) = delete;
    EventPool& operator = (const EventPool&) = delete;
    EventPool& operator = (EventPool&&) = delete;

    ~EventPool() = default;

    void release() {
        std::lock_guard<mutex_t> _(event_mutex_);
        for (auto& event : event_pool_) {
            deleter_(event);
            allocate_num_--;
        }
        event_pool_.clear();
    }

    void get(T& event) {
        std::lock_guard<mutex_t> _(event_mutex_);
        if (event_pool_.empty()) {
            allocator_(event);
            allocate_num_++;
        } else {
            event = event_pool_.back();
            event_pool_.pop_back();
        }
    }

    void restore(T& event) {
        std::lock_guard<mutex_t> _(event_mutex_);
        event_pool_.emplace_back(event);
    }
};

// GlobalEventPool
static EventPool<deviceEvent_t> gDIPUEventPool(
    [](deviceEvent_t& event) {
        devapis::createEvent(&event);
    }, [](deviceEvent_t& event) {
        devapis::destroyEvent(event);
    }
);


void getEventFromPool(deviceEvent_t& event) {
    gDIPUEventPool.get(event);
}

void restoreEventToPool(deviceEvent_t& event) {
    gDIPUEventPool.restore(event);
}
void releaseGlobalEventPool() {
    gDIPUEventPool.release();
}

}  // namespace dipu
