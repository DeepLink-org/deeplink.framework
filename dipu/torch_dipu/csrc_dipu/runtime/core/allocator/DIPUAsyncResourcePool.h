// Copyright (c) 2023, DeepLink.
#pragma once

#include <deque>
#include <mutex>
#include <tuple>

#include "csrc_dipu/runtime/core/DIPUEvent.h"

namespace dipu {

constexpr size_t kMaxAsyncResourcePoolLength = 32;

template <class T>
class AsyncResourcePool {
 public:
  virtual void add(const T& t, std::deque<DIPUEvent>& events) = 0;
  virtual T get() = 0;
  virtual bool ready() const = 0;
  virtual bool empty() const = 0;
  virtual size_t size() const = 0;
};

template <class T, at::DeviceType device_type, int algorithm>
class AsyncResourcePoolImpl : public AsyncResourcePool<T> {
  using Res = std::tuple<T, std::deque<DIPUEvent>>;
  std::deque<Res> list_;
  using mutex_t = std::mutex;
  mutable mutex_t mutex_;

 public:
  void add(const T& t, std::deque<DIPUEvent>& events) override {
    std::lock_guard<mutex_t> lk(mutex_);
    if (events.size() > 0) {
      list_.emplace_back(t, std::move(events));
    } else {
      list_.emplace_front(t, std::move(events));
    }
  }

  T get() override {
    std::lock_guard<mutex_t> lk(mutex_);
    T t = std::get<0>(list_.front());
    list_.pop_front();
    return t;
  }

  bool empty() const override {
    std::lock_guard<mutex_t> lk(mutex_);
    return list_.empty();
  }

  bool ready() const override {
    std::lock_guard<mutex_t> lk(mutex_);
    if (list_.empty()) {
      return false;
    }

    for (auto& item : std::get<1>(list_.front())) {
      if (!item.query()) {
        return false;
      }
    }

    return true;
  }

  size_t size() const override {
    std::lock_guard<mutex_t> lk(mutex_);
    return list_.size();
  }
};

}  // namespace dipu
