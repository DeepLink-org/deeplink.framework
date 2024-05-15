// Copyright (c) 2023, DeepLink.
#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <tuple>

#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>

#include "csrc_dipu/runtime/core/DIPUEvent.h"

namespace dipu {

template <class T>
class AsyncResourcePool {
 public:
  virtual void add(const T& t, std::deque<DIPUEvent>& events) = 0;
  virtual T get() = 0;
  virtual bool ready() = 0;
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
    if (events.empty()) {
      list_.emplace_front(t, std::move(events));
    } else {
      list_.emplace_back(t, std::move(events));
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

  bool ready() override {
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

template <class T, at::DeviceType device_type, int algorithm>
class AsyncResourceMultiStreamPoolImpl : public AsyncResourcePool<T> {
 private:
  struct Resource {
    const T t;
    int event_count;
    const std::deque<DIPUEvent> events;

    Resource(const T& t, int event_count, std::deque<DIPUEvent>& events)
        : t(t), event_count(event_count), events(std::move(events)) {}
  };
  ska::flat_hash_map<c10::StreamId,
                     std::queue<std::pair<std::shared_ptr<Resource>, int>>>
      queues;

  std::shared_ptr<Resource> ready_resource;

  using mutex_t = std::mutex;
  mutable mutex_t mutex;

 public:
  void add(const T& t, std::deque<DIPUEvent>& events) override {
    std::lock_guard<mutex_t> lk(mutex);
    auto resource = std::make_shared<Resource>(t, events.size(), events);
    if (0 == resource->event_count) {
      // Special queue for resources that have no events to wait
      queues[-1].push({resource, -1});
    } else {
      for (int i = 0; i < resource->event_count; ++i) {
        queues[resource->events[i].stream_id()].push({resource, i});
      }
    }
  }

  T get() override {
    std::lock_guard<mutex_t> lk(mutex);
    TORCH_CHECK(ready_resource, "No ready resource to get!")
    T t = ready_resource->t;
    ready_resource.reset();
    return t;
  }

  bool empty() const override {
    std::lock_guard<mutex_t> lk(mutex);
    return queues.empty();
  }

  // Remove completed events in queues and decrease event_count until a resource
  // with event_count of 0 is found, then save it to ready_resource
  bool ready() override {
    std::lock_guard<mutex_t> lk(mutex);

    if (ready_resource) {
      return true;
    }

    for (auto it = queues.begin(); it != queues.end();) {
      auto& [stream_id, queue] = *it;
      auto& [resource, event_id] = queue.front();
      if (0 == resource->event_count || resource->events[event_id].query()) {
        // Skip --event_count for resources in the special queue
        // since their event_count is already 0
        if (resource->event_count > 0) {
          --resource->event_count;
        }
        if (0 == resource->event_count) {
          ready_resource = resource;
        }
        queue.pop();
        if (queue.empty()) {
          it = queues.erase(it);
        } else {
          ++it;
        }
        if (ready_resource) {
          return true;
        }
      }
    }

    return false;
  }

  size_t size() const override {
    std::lock_guard<mutex_t> lk(mutex);
    size_t size_sum = 0;
    for (auto& [stream_id, queue] : queues) {
      size_sum += queue.size();
    }
    return size_sum;
  }
};

}  // namespace dipu
