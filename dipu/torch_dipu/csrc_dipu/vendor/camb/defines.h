// Copyright (c) 2023, DeepLink.
#pragma once

#include <cnpapi.h>

#define DIPU_CALLCNPAPI(Expr)                                               \
  do {                                                                      \
    cnpapiResult ret = Expr;                                                \
    TORCH_CHECK(ret == CNPAPI_SUCCESS, "call cnpapi error, expr = ", #Expr, \
                ", ret = ", ret);                                           \
  } while (0)
