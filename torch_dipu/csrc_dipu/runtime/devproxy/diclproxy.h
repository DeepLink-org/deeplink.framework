// Copyright (c) 2023, DeepLink.
#pragma once

#include "../device/diclapis.h"

using dipu::devapis::diclResult_t;
using dipu::devapis::ReduceOp;
namespace dipu {

// need enhance return status.
namespace devproxy {

  DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm);

  DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId);

  DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId, int rank, int localDeviceId = -1);

  // DIPU_API void diclCommInitAll(diclComm_t* comms, int ndev, const int* devlist);

  DIPU_API diclResult_t diclCommDestroy(diclComm_t comm);

  // DIPU_API diclResult_t diclCommFinalize(diclComm_t comm);

  // DIPU_API diclResult_t diclCommAbort(diclComm_t comm);

  DIPU_API diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                              const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclBroadcast(const void *sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                              int root, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclAllGather(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                              diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclReduceScatter(void *sendbuff, void *recvbuff, uint64_t count, at::ScalarType datatype, 
                                  const ReduceOp& op, diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclSend(void* sendbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream);

  DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream);


} // namespace devproxy

} // namespace dipu