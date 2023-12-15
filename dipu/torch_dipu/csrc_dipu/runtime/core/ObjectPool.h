// Copyright (c) 2023, DeepLink.
#pragma once

#include <mutex>
#include <queue>

namespace dipu {

constexpr int kDefaultMaxObjectSize(100000);

template <typename T>
class ObjectPool final {
 public:
  T* allocate() {
    T* t = nullptr;
    {
      std::lock_guard<std::mutex> lck(mtx_);
      if (!queue_.empty()) {
        t = queue_.front();
        queue_.pop();
      }
    }

    if (t == nullptr) {
      t = new T();
    }
    return t;
  }

  void free(T* t) {
    bool full = false;
    t->~T();
    {
      std::lock_guard<std::mutex> lck(mtx_);
      if (queue_.size() < max_object_size_) {
        queue_.push(t);
      } else {
        full = true;
      }
    }

    if (full) {
      new (t) T();
      delete t;
    }
  }

  ~ObjectPool() {
    while (!queue_.empty()) {
      T* t = queue_.front();
      new (t) T();
      queue_.pop();
      delete t;
    }
  }

 private:
  std::mutex mtx_;
  std::queue<T*> queue_;
  int32_t max_object_size_ = kDefaultMaxObjectSize;
};

}  // namespace dipu
