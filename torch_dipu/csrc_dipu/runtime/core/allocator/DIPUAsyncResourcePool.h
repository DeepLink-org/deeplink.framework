// Copyright (c) 2023, DeepLink.
#pragma once

#include <mutex>
#include <deque>
#include <tuple>
#include <deque>
#include "DIPUSpinMutex.h"
#include "../DIPUEvent.h"

namespace dipu {

template<class T>
class AsyncResourcePool {
public:
    virtual void add(const T& t, std::deque<DIPUEvent>& events) = 0;
    virtual T get() = 0;
    virtual bool ready() = 0;
    virtual size_t size() = 0;
};


template<class T, at::DeviceType device_type, int algorithm>
class AsyncResourcePoolImpl: public AsyncResourcePool<T>{
};

template<class T, int algorithm>
class AsyncResourcePoolImpl<T, at::DeviceType::CPU, algorithm>: public AsyncResourcePool<T>{
  std::deque<T> list_;
  using mutex_t = std::mutex;
  mutex_t mutex_;
  public:
    void add(const T& t, std::deque<DIPUEvent>& events) override {
      std::lock_guard<mutex_t> lk(mutex_);
      list_.push_back(t);
    }

    T get() override {
      std::lock_guard<mutex_t> lk(mutex_);
      T t = list_.front();
      list_.pop_front();
      return t;
    }

    bool ready() override {
      std::lock_guard<mutex_t> lk(mutex_);
      return !list_.empty();
    }

    size_t size() override {
      std::lock_guard<mutex_t> lk(mutex_);
      return list_.size();
    }
};

template<class T, int algorithm>
class AsyncResourcePoolImpl<T, dipu::DIPU_DEVICE_TYPE, algorithm> : public AsyncResourcePool<T>{
    using Res = std::tuple<T, std::deque<DIPUEvent>>;
    std::deque<Res> list_;
    using mutex_t = std::mutex;
    mutex_t mutex_;
  public:
    void add(const T& t, std::deque<DIPUEvent>& events) override {
      std::lock_guard<mutex_t> lk(mutex_);
      list_.emplace_back(t, std::move(events));
    }

    T get() override {
      std::lock_guard<mutex_t> lk(mutex_);
      T t = std::get<0>(list_.front());
      list_.pop_front();
      return t;
    }

    bool ready() override {
      std::lock_guard<mutex_t> lk(mutex_);
      if (list_.empty()) {
        return false;
      }

      for (auto iter = std::get<1>(list_.front()).begin(); iter != std::get<1>(list_.front()).end(); iter++) {
        if (iter->query() == false) {
          return false;
        }
      }
      return true;
    }

    size_t size() override {
      std::lock_guard<mutex_t> lk(mutex_);
      return list_.size();
    }
};

} // namespace dipu