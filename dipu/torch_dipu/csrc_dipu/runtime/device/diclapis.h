#pragma once

#include <c10/core/ScalarType.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <csrc_dipu/vendor/vendorapi.h>

#include "./deviceapis.h"

namespace dipu {

// need add return status.
namespace devapis {
// todo: define new diopi reduceop.
using ReduceOp = c10d::ReduceOp;

extern const int DICL_UNIQUE_ID_BYTES_SIZE;

// todo:: dipu only export devproxy but not devapis (which move o diopi)
DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm);

DIPU_API diclResult_t diclGetUniqueId(commUniqueId *uniqueId);

DIPU_API diclResult_t diclCommInitRank(diclComm_t *comm, int nranks,
                                       commUniqueId uniqueId, int rank,
                                       int localDeviceId = -1);

// DIPU_API void diclCommInitAll(diclComm_t* comms, int ndev, const int*
// devlist);

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm);

// DIPU_API diclResult_t diclCommFinalize(diclComm_t comm);

// DIPU_API diclResult_t diclCommAbort(diclComm_t comm);

DIPU_API diclResult_t diclAllReduce(const void *sendBuf, void *recvBuf,
                                    size_t count, at::ScalarType datatype,
                                    const ReduceOp &reduceOp, diclComm_t comm,
                                    deviceStream_t stream);

DIPU_API diclResult_t diclBroadcast(const void *sendBuf, void *recvBuf,
                                    size_t count, at::ScalarType datatype,
                                    int root, diclComm_t comm,
                                    deviceStream_t stream);

DIPU_API diclResult_t diclAllGather(const void *sendBuf, void *recvBuf,
                                    size_t sendCount, at::ScalarType datatype,
                                    diclComm_t comm, deviceStream_t stream);

DIPU_API diclResult_t diclReduce(const void *sendbuf, void *recvBuf,
                                 size_t count, at::ScalarType datatype,
                                 const ReduceOp &reduceOp, int root,
                                 diclComm_t comm, deviceStream_t stream);

DIPU_API diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf,
                                        size_t recvCount,
                                        at::ScalarType datatype,
                                        const ReduceOp &op, diclComm_t comm,
                                        deviceStream_t stream);

DIPU_API diclResult_t diclSend(void *recvBuf, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream);

DIPU_API diclResult_t diclRecv(void *recvBuf, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream);

}  // namespace devapis

}  // namespace dipu