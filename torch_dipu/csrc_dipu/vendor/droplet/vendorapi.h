
#pragma once
#include <tang_runtime.h>

#include <csrc_dipu/common.h>

namespace dipu {


#define DIPU_CALLDROPLET(Expr)   {                                                     \
    tangError_t ret = Expr;                                                         \
    if (ret != tangSuccess) {                                                       \
        printf("call a tangrt function (%s) failed. return code=%d", #Expr, ret);   \
        throw std::runtime_error("dipu device error");                              \
    } \
}

using deviceStream_t = tangStream_t;
#define deviceDefaultStreamLiteral nullptr
using deviceEvent_t = tangEvent_t;

}





