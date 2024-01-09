// Copyright (c) 2023, DeepLink.
#pragma once

#include <cstdint>

namespace dipu {

enum class NativeMemoryFormat_t : int64_t {
  UNDEFINED = -1,
  NCHW = 0,
  NHWC = 1,
  ND = 2,
  NC1HWC0 = 3,
  FRACTAL_Z = 4,
  NC1HWC0_C04 = 12,
  HWCN = 16,
  NDHWC = 27,
  FRACTAL_NZ = 29,
  NCDHW = 30,
  NDC1HWC0 = 32,
  FRACTAL_Z_3D = 33
};

}  // namespace dipu
