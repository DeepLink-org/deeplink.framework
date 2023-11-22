#include "DIPUEventPool.h"

#include <deque>
#include <functional>
#include <iostream>
#include <mutex>

namespace dipu {

template <typename T>
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
      : allocator_(allocator), deleter_(deleter) {}

  EventPool(const EventPool&) = delete;
  EventPool(EventPool&&) = delete;
  EventPool& operator=(const EventPool&) = delete;
  EventPool& operator=(EventPool&&) = delete;

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
    bool need_allocator = false;
    {
      std::lock_guard<mutex_t> _(event_mutex_);
      if (event_pool_.empty()) {
        need_allocator = true;
      } else {
        event = event_pool_.back();
        event_pool_.pop_back();
      }
    }
    if (need_allocator) {
      allocator_(event);
    }
  }

  void restore(T& event) {
    std::lock_guard<mutex_t> _(event_mutex_);
    event_pool_.emplace_back(event);
  }
};

EventPool<deviceEvent_t>* getEventPool() {
  const int index = devproxy::current_device();
// GlobalEventPool for different cards , construct when really needed
#define dispatch_event_pool(device_id)                               \
  if (index == device_id) {                                          \
    static EventPool<deviceEvent_t> gDIPUEventPool(                  \
        [](deviceEvent_t& event) { devapis::createEvent(&event); },  \
        [](deviceEvent_t& event) { devapis::destroyEvent(event); }); \
    return &gDIPUEventPool;                                          \
  }

  dispatch_event_pool(0);
  dispatch_event_pool(1);
  dispatch_event_pool(2);
  dispatch_event_pool(3);
  dispatch_event_pool(4);
  dispatch_event_pool(5);
  dispatch_event_pool(6);
  dispatch_event_pool(7);
  dispatch_event_pool(8);
  dispatch_event_pool(9);
  dispatch_event_pool(10);
  dispatch_event_pool(11);
  dispatch_event_pool(12);
  dispatch_event_pool(13);
  dispatch_event_pool(14);
  dispatch_event_pool(15);
  TORCH_CHECK(false, "support up to 16 cards");
}

void getEventFromPool(deviceEvent_t& event) { getEventPool()->get(event); }

void restoreEventToPool(deviceEvent_t& event) {
  getEventPool()->restore(event);
}

void releaseAllEvent() { getEventPool()->release(); }

}  // namespace dipu
