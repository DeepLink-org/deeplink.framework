// Copyright (c) 2023, DeepLink.
#pragma once

#include <string>

#include "./basedef.h"

namespace dipu {
namespace devapis {

DIPU_WEAK void enableProfiler(const std::string &dump_path, bool call_stack, bool record_shapes, bool profile_memory);
DIPU_WEAK void disableProfiler();

}  // end namespace devapis
}  // end namespace dipu