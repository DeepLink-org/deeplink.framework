// Copyright (c) 2024, DeepLink.
#pragma once

#include "distributedUtil.h"

namespace dipu {

class UtilInstance {
 public:
  UtilInstance(const UtilInstance&) = delete;
  UtilInstance& operator=(const UtilInstance&) = delete;

  static UtilInstance& getInstance() {
    static UtilInstance instance;
    return instance;
  }

  // get current device special impl object.
  distributedUtil* getVendorImpl();

  // if not impl, will add a default impl.
  void setVendorImpl(distributedUtil* obj);

 private:
  UtilInstance() = default;
  ~UtilInstance() = default;

  distributedUtil* util_ = nullptr;
};

}  // namespace dipu
