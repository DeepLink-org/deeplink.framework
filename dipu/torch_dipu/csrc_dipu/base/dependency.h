// Copyright (c) 2023, DeepLink.
#pragma once

#include <torch/version.h>

#if TORCH_VERSION_MINOR < 100 && TORCH_VERSION_PATCH < 100
#define DIPU_TORCH_VERSION                                   \
  ((TORCH_VERSION_MAJOR * 100 + TORCH_VERSION_MINOR) * 100 + \
   TORCH_VERSION_PATCH)
#else
#error "require refactoring: version number exceeds limit"
#endif
