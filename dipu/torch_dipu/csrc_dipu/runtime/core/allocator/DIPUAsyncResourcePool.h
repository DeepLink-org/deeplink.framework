// Copyright (c) 2023, DeepLink.
#pragma once

#include <deque>
#include <iostream>
#include <mutex>
#include <queue>
#include <tuple>
#include <utility>

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

#define OneStreamOneQueueAlgo 1

// This implementation provides a separate queue for each stream
template <class T, at::DeviceType device_type>
class AsyncResourcePoolImpl<T, device_type, OneStreamOneQueueAlgo>
    : public AsyncResourcePool<T> {
 private:
  // Resources that have events to wait for
  struct Resource final {
    const T t;
    int event_count;

    Resource(const T& t, int event_count) : t(t), event_count(event_count) {}

    ~Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&&) = delete;
    Resource& operator=(Resource&&) = delete;
  };
  ska::flat_hash_map<c10::StreamId, std::queue<std::pair<std::shared_ptr<Resource>, DIPUEvent>>>
      queues_with_events;
  std::shared_ptr<Resource> ready_resource ;

  // Resources that have no events to wait for.
  // In other words, they are already ready.
  // Place them in a special queue for higher performance.
  std::queue<T> queue_without_events;

  size_t total_size = 0;

  using mutex_t = std::mutex;
  mutable mutex_t mutex;

 public:
  void add(const T& t, std::deque<DIPUEvent>& events) override {
    std::lock_guard<mutex_t> lk(mutex);
    if (events.empty()) {
      queue_without_events.push(t);
    } else {
      auto resource = std::make_shared<Resource>(t, events.size());
      std::cout << "Add resource: " << std::get<1>(resource->t) << std::endl;
      for (DIPUEvent& event : events) {
        queues_with_events[event.stream_id()].emplace(resource,
                                                      std::move(event));
      }
      events.clear();
    }
    ++total_size;
  }

  T get() override {
    std::lock_guard<mutex_t> lk(mutex);

    if (!queue_without_events.empty()) {
      T t = queue_without_events.front();
      queue_without_events.pop();
      --total_size;
      return t;
    }

    TORCH_CHECK(ready_resource, "No ready resource to get!")
    T t = ready_resource->t;
    std::cout << "get resource: " << std::get<1>(t) << std::endl;
    // delete ready_resource;
    // ready_resource = nullptr;
    ready_resource.reset();
    --total_size;
    return t;
  }

  bool empty() const override {
    std::lock_guard<mutex_t> lk(mutex);
    return 0 == total_size;
  }

  // If there is no ready resource, remove completed events in queues
  // and decrease event_count until a resource with event_count of 0 is found,
  // then save it as a ready resource
  bool ready() override {
    std::lock_guard<mutex_t> lk(mutex);

    if (!queue_without_events.empty()) {
      return true;
    }

    if (ready_resource) {
      return true;
    }

    for (auto it = queues_with_events.begin();
         it != queues_with_events.end();) {
      auto& [stream_id, queue] = *it;
      auto& [resource, event] = queue.front();
      if (event.query()) {
        --resource->event_count;
        if (0 == resource->event_count) {
  	  std::cout << "Ready resource: " << std::get<1>(resource->t) << std::endl;
          ready_resource = resource;
        }
        queue.pop();
      }

      if (queue.empty()) {
        it = queues_with_events.erase(it);
      } else {
        ++it;
      }

      if (ready_resource) {
        return true;
      }
    }

    return false;
  }

  size_t size() const override {
    std::lock_guard<mutex_t> lk(mutex);
    return total_size;
  }
};

}  // namespace dipu
