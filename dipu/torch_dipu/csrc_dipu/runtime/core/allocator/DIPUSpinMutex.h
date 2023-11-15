// Copyright (c) 2023, DeepLink.
#pragma once

#include <atomic>
#include <thread>

namespace dipu {

/// Simple spin-lock to help build thread-safe functions.
class SpinMutex {
 private:
  std::atomic<bool> excl_{false};

 public:
  constexpr SpinMutex() noexcept = default;

  SpinMutex(const SpinMutex &) = delete;

  void delay() const noexcept { std::this_thread::yield(); }

  void lock() {
    for (bool exp = false;
         !excl_.compare_exchange_weak(exp, true, std::memory_order_acq_rel);
         exp = false)
      delay();
  }

  bool try_lock() {
    bool exp = false;
    return excl_.compare_exchange_weak(exp, true, std::memory_order_acq_rel);
  }

  void unlock() { excl_.store(false, std::memory_order_release); }
};

}  // namespace dipu
