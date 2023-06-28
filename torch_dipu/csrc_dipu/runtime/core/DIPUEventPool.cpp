#include "DIPUEventPool.h"
#include <list>
#include <mutex>
#include <functional>


namespace dipu {

template<typename T>
class EventPool final {
protected:
    std::list<T> eventPool_;

    std::function<void(T&)> allocator_;
    std::function<void(T&)> deleter_;
    std::mutex eventMtx_;

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
        std::unique_lock<std::mutex> _(eventMtx_);
        for (auto& event : eventPool_) {
            deleter_(event);
        }
        eventPool_.clear();
    }

    void get(T& event) {
        std::unique_lock<std::mutex> _(eventMtx_);
        if (eventPool_.empty()) {
            allocator_(event);
        } else {
            event = eventPool_.back();
            eventPool_.pop_back();
        }
    }

    void restore(T& event) {
        std::unique_lock<std::mutex> _(eventMtx_);
        eventPool_.emplace_back(event);
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
