// Copyright (c) 2024, DeepLink.
#pragma once

#include "distributedUtil.h"

namespace dipu {

class Singleton {
 public:
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

  static Singleton& getInstance() {
    static Singleton instance;
    return instance;
  }

  distributedUtil* getUtil();

  void setUtilObj(distributedUtil* obj);

 private:
  Singleton() = default;
  ~Singleton() = default;

  distributedUtil* util_ = nullptr;
};

}  // namespace dipu
