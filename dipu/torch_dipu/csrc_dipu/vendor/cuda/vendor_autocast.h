// Copyright (c) 2023, DeepLink.
//
// Vendor-specific AMP autocast configuration file.

#pragma once

// See this included file for more details.
#include "csrc_dipu/aten/ops/DIPUAmp.hpp"

namespace dipu {
namespace autocast {

// Uncomment the next line to make test_autocast.py FAIL.
// DIPU_CUSTOMIZE_OP_CAST_POLICY(mm, kFp32);

}  // namespace autocast
}  // namespace dipu
