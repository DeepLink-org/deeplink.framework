// Copyright (c) 2023, DeepLink.
#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "csrc_dipu/runtime/core/DIPUEvent.h"

namespace dipu {

constexpr size_t kMaxAsyncResourcePoolLength = 3;

template <typename T>
class AsyncResourceQueue {
  using value_type = std::pair<T, std::vector<DIPUEvent>>;

  std::deque<value_type> queue;
  mutable std::mutex mutex;
  std::atomic_size_t count{};

 public:
  void put(T&& item, std::vector<DIPUEvent>&& events) {
    std::scoped_lock _(mutex);
    queue.emplace_back(std::move(item), std::move(events));
    count.store(queue.size(), std::memory_order_release);
  }

  std::optional<T> pop() {
    if (empty()) {
      return {};
    }

    std::scoped_lock _(mutex);
    if (queue.empty()) {
      return {};
    }

    auto& front = queue.front();
    for (auto& event : front.second) {
      if (!event.query()) {
        return {};
      }
    }

    auto item = std::move(front.first);
    queue.pop_front();
    count.store(queue.size(), std::memory_order_release);
    return item;
  }

  bool empty() const { return count.load(std::memory_order_acquire) != 0; }
  std::size_t size() const { return count.load(std::memory_order_acquire); }
};

}  // namespace dipu
