// Copyright (c) 2023, DeepLink.
#pragma once

#include <mutex>
#include <deque>
#include <tuple>
#include "DIPUEvent.h"

namespace dipu {

template<class T>
class AsyncResoursePool {
public:
    virtual void add(const T& t) = 0;
    virtual T get() = 0;
    virtual bool ready() = 0;
    virtual size_t size() = 0;
};


template<class T, at::DeviceType device_type, int algorithm>
class AsyncResoursePoolImpl: public AsyncResoursePool<T>{
};

template<class T, int algorithm>
class AsyncResoursePoolImpl<T, at::DeviceType::CPU, algorithm>: public AsyncResoursePool<T>{
  std::deque<T> list_;
  using mutex_t = std::mutex;
  mutex_t mutex_;
  public:
    void add(const T& t) override {
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
        return list_.size();
    }
};

template<class T, int algorithm>
class AsyncResoursePoolImpl<T, dipu::DIPU_DEVICE_TYPE, algorithm> : public AsyncResoursePool<T>{
    using Res = std::tuple<T, DIPUEvent>;
    std::deque<Res> list_;
    using mutex_t = std::mutex;
    mutex_t mutex_;
  public:
    void add(const T& t) override {
      std::lock_guard<mutex_t> lk(mutex_);
      list_.emplace_back(t, DIPUEvent());
      std::get<1>(list_.back()).record();
    }

    T get() override {
        std::lock_guard<mutex_t> lk(mutex_);
        T t = std::get<0>(list_.front());
        list_.pop_front();
        return t;
    }

    bool ready() override {
      std::lock_guard<mutex_t> lk(mutex_);
      return (!list_.empty()) && (std::get<1>(list_.front())).query();
    }

    size_t size() override {
        return list_.size();
    }
};

} // namespace dipu