// Copyright (c) 2023, DeepLink.
#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace dipu {

constexpr size_t kMaxAsyncResourcePoolLength = 3;

template <typename T, typename U>
class AsyncResourceQueue {
  std::deque<std::pair<T, U>> queue;
  std::mutex mutex;
  std::atomic_size_t count{};

 public:
  auto put(T&& item, U&& ready) -> void {
    std::scoped_lock _(mutex);
    queue.emplace_back(std::move(item), std::move(ready));
    count.store(queue.size(), std::memory_order_release);
  }

  auto pop() -> std::optional<T> {
    if (empty()) {
      return {};
    }

    std::scoped_lock _(mutex);
    if (queue.empty()) {
      return {};
    }

    auto& [item, ready] = queue.front();
    if (not ready()) {
      return {};
    }

    auto output = std::move(item);
    queue.pop_front();
    count.store(queue.size(), std::memory_order_release);
    return output;
  }

  auto size() const -> std::size_t {
    return count.load(std::memory_order_acquire);
  }

  auto empty() const -> bool {
    return count.load(std::memory_order_acquire) == 0;
  }
};

}  // namespace dipu
