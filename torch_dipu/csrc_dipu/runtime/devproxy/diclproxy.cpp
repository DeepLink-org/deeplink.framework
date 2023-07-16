// Copyright (c) 2023, DeepLink.

#include "./diclproxy.h"

namespace dipu {

// need enhance return status.
namespace devproxy {

  diclResult_t diclGetCommAsyncError(diclComm_t comm) {
    devapis::diclGetCommAsyncError(comm);
  }

  diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
    devapis::diclGetUniqueId(uniqueId);
  }

  diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId, int rank, int localDeviceId) {
    devapis::diclCommInitRank(comm, nranks, uniqueId, rank, localDeviceId);
  }

  // void diclCommInitAll(diclComm_t* comms, int ndev, const int* devlist);

  diclResult_t diclCommDestroy(diclComm_t comm) {
    devapis::diclCommDestroy(comm);
  }

  //  diclResult_t diclCommFinalize(diclComm_t comm);

  //  diclResult_t diclCommAbort(diclComm_t comm);

  diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                              const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream)  {
    devapis::diclAllReduce(sendbuff, recvbuff, count, datatype, reduceOp, comm, stream);
  }

  diclResult_t diclBroadcast(const void *sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                              int root, diclComm_t comm, deviceStream_t stream) {
    devapis::diclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
  }

  diclResult_t diclAllGather(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                              diclComm_t comm, deviceStream_t stream) {
    devapis::diclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
  }

  diclResult_t diclReduce(const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream)  {
    devapis::diclReduce(sendbuff, recvbuff, count, datatype, reduceOp, root, comm, stream);
  }

  diclResult_t diclReduceScatter(void *sendbuff, void *recvbuff, uint64_t count, at::ScalarType datatype, 
                                  const ReduceOp& op, diclComm_t comm, deviceStream_t stream) {
    devapis::diclReduceScatter(sendbuff, recvbuff, count, datatype, op, comm, stream);
  }

  diclResult_t diclSend(void* sendbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream) {
    devapis::diclSend(sendbuff, count, datatype, peer, comm, stream);
  }

  diclResult_t diclRecv(void* recvbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream) {
    devapis::diclRecv(recvbuff, count, datatype, peer, comm, stream);
  }


} // namespace devproxy

} // namespace dipu