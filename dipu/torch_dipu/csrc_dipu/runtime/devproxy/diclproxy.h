// Copyright (c) 2023, DeepLink.
#pragma once

#include "csrc_dipu/runtime/device/diclapis.h"

namespace dipu {

#define DIPU_CALL_DICLAPIS(Expr)                                        \
  {                                                                     \
    devapis::diclResult_t ret = Expr;                                   \
    TORCH_CHECK(ret == devapis::diclResult_t::DICL_SUCCESS,             \
                "call diclapis error, expr = ", #Expr, ", ret = ", ret) \
  }

// need enhance return status.
namespace devproxy {

DIPU_API devapis::diclResult_t diclGetCommAsyncError(diclComm_t comm);

DIPU_API devapis::diclResult_t diclGetUniqueId(commUniqueId* uniqueId);

DIPU_API devapis::diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
                                                commUniqueId uniqueId, int rank,
                                                int localDeviceId = -1);

DIPU_API devapis::diclResult_t diclCommDestroy(diclComm_t comm);

DIPU_API devapis::diclResult_t diclAllReduce(
    const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
    const devapis::ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream);

DIPU_API devapis::diclResult_t diclBroadcast(const void* sendbuff,
                                             void* recvbuff, size_t count,
                                             at::ScalarType datatype, int root,
                                             diclComm_t comm,
                                             deviceStream_t stream);

DIPU_API devapis::diclResult_t diclAllGather(const void* sendbuff,
                                             void* recvbuff, size_t sendCount,
                                             at::ScalarType datatype,
                                             diclComm_t comm,
                                             deviceStream_t stream);

DIPU_API devapis::diclResult_t diclGather(void* sendbuff, void** recvbuff,
                                          size_t count, at::ScalarType datatype,
                                          int root, int curRank, int numRanks,
                                          diclComm_t comm,
                                          deviceStream_t stream);

DIPU_API devapis::diclResult_t diclScatter(void** sendbuff, void* recvbuff,
                                           size_t count,
                                           at::ScalarType datatype, int root,
                                           int curRank, int numRanks,
                                           diclComm_t comm,
                                           deviceStream_t stream);

DIPU_API devapis::diclResult_t diclReduce(const void* sendbuff, void* recvbuff,
                                          size_t count, at::ScalarType datatype,
                                          const devapis::ReduceOp& reduceOp,
                                          int root, diclComm_t comm,
                                          deviceStream_t stream);

DIPU_API devapis::diclResult_t diclReduceScatter(
    void* sendbuff, void* recvbuff, size_t recvCount, at::ScalarType datatype,
    const devapis::ReduceOp& op, diclComm_t comm, deviceStream_t stream);

DIPU_API devapis::diclResult_t diclSend(void* sendbuff, size_t count,
                                        at::ScalarType datatype, int peer,
                                        diclComm_t comm, deviceStream_t stream);

DIPU_API devapis::diclResult_t diclRecv(void* recvbuff, size_t count,
                                        at::ScalarType datatype, int peer,
                                        diclComm_t comm, deviceStream_t stream);

DIPU_WEAK devapis::diclResult_t diclGetCommName(std::string& commName,
                                                diclComm_t comm);

}  // namespace devproxy
}  // namespace dipu
