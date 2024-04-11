// Copyright (c) 2024, DeepLink.

#include "UtilInstance.h"

namespace dipu {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static distributedUtil default_impl_;

distributedUtil* UtilInstance::getVendorImpl() {
  if (!util_) {
    util_ = &default_impl_;
  }
  return util_;
}

void UtilInstance::setVendorImpl(distributedUtil* obj) { util_ = obj; }

}  // namespace dipu
