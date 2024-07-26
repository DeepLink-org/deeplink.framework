#include "DIPUEventPool.h"

#include <array>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>

#include "csrc_dipu/base/utility.h"
#include "csrc_dipu/runtime/device/deviceapis.h"

namespace {

class EventPool {
  std::deque<dipu::deviceEvent_t> pool;
  std::recursive_mutex mutex;

 public:
  void acquire(dipu::deviceEvent_t& event) {
    {
      std::scoped_lock _(mutex);
      if (!pool.empty()) {
        event = pool.back();
        pool.pop_back();
        return;
      }
    }

    dipu::devapis::createEvent(&event);
  }

  void release(dipu::deviceEvent_t& event) {
    std::scoped_lock _(mutex);
    pool.emplace_back(event);
  }

  void clear() {  // should it called inside destructor?
    std::scoped_lock _(mutex);
    for (auto& event : pool) {
      dipu::devapis::destroyEvent(event);
    }
    pool.clear();
  }
};

struct EventPoolHolder {
  template <std::size_t I>
  inline static auto& value() {
    auto static instance = EventPool();
    return instance;
  }
};

auto constexpr max_card_number = 16;
using EventPoolHolderArray =
    dipu::static_function_array<EventPoolHolder, max_card_number>;

auto event_pool(int index) -> EventPool& {
  TORCH_CHECK(0 <= index and index < max_card_number, "support up to 16 cards");
  return EventPoolHolderArray::value[index]();
}

}  // namespace

namespace dipu {

void event_pool_acquire(int index, deviceEvent_t& event) {
  event_pool(index).acquire(event);
}

void event_pool_release(int index, deviceEvent_t& event) {
  event_pool(index).release(event);
}

void event_pool_clear() {
  for (auto& pool : EventPoolHolderArray::value) {
    pool().clear();
  }
}

}  // namespace dipu
