#include "DIPUEventPool.h"

#include <deque>
#include <mutex>

#include "csrc_dipu/base/basedef.h"
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

struct StaticEventPoolArray
    : dipu::static_value_array<StaticEventPoolArray, dipu::kMaxDeviceNumber> {
  template <std::size_t I>
  inline static auto& value() {
    auto static instance = EventPool();
    return instance;
  }
};

auto event_pool(int index) -> EventPool& {
  TORCH_CHECK(0 <= index and index < dipu::kMaxDeviceNumber,
              "device index out of range");
  return StaticEventPoolArray::get(index);
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
  for (auto& getter : StaticEventPoolArray::array()) {
    getter().clear();
  }
}

}  // namespace dipu
