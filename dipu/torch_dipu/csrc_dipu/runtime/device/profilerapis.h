// Copyright (c) 2023, DeepLink.
#pragma once

#include <string>

#include "./basedef.h"

namespace dipu {
namespace devapis {

DIPU_WEAK void enableProfiler(const std::string &dump_path, bool call_stack);
DIPU_WEAK void disableProfiler();

}  // end namespace devapis
}  // end namespace dipu