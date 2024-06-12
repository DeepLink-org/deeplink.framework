// Copyright (c) 2023, DeepLink.

#include "diclproxy.h"

namespace dipu {
// need enhance return status.
namespace devproxy {

devapis::diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  return devapis::diclGetCommAsyncError(comm);
}

devapis::diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  return devapis::diclGetUniqueId(uniqueId);
}

devapis::diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
                                       commUniqueId uniqueId, int rank,
                                       int localDeviceId) {
  return devapis::diclCommInitRank(comm, nranks, uniqueId, rank, localDeviceId);
}

devapis::diclResult_t diclCommDestroy(diclComm_t comm) {
  return devapis::diclCommDestroy(comm);
}

devapis::diclResult_t diclAllReduce(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    const devapis::ReduceOp& reduceOp,
                                    diclComm_t comm, deviceStream_t stream) {
  return devapis::diclAllReduce(sendbuff, recvbuff, count, datatype, reduceOp,
                                comm, stream);
}

devapis::diclResult_t diclBroadcast(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    int root, diclComm_t comm,
                                    deviceStream_t stream) {
  return devapis::diclBroadcast(sendbuff, recvbuff, count, datatype, root, comm,
                                stream);
}

devapis::diclResult_t diclAllGather(const void* sendbuff, void* recvbuff,
                                    size_t sendCount, at::ScalarType datatype,
                                    diclComm_t comm, deviceStream_t stream) {
  return devapis::diclAllGather(sendbuff, recvbuff, sendCount, datatype, comm,
                                stream);
}

devapis::diclResult_t diclGather(void* sendbuf, void* const* recvbuf,
                                 size_t count, at::ScalarType datatype,
                                 int root, int curRank, int numRanks,
                                 diclComm_t comm, deviceStream_t stream) {
  if (curRank != root) {
    DIPU_CALL_DICLAPIS(diclSend(sendbuf, count, datatype, root, comm, stream));
    return devapis::diclResult_t::DICL_SUCCESS;
  }

  for (const auto srcRank : c10::irange(numRanks)) {
    if (srcRank == root) {
      continue;
    }
    DIPU_CALL_DICLAPIS(
        diclRecv(recvbuf[srcRank], count, datatype, srcRank, comm, stream));
  }

  auto deviceId = static_cast<devapis::deviceId_t>(curRank);
  devapis::memCopyD2DAsync(stream, count * c10::elementSize(datatype), deviceId,
                           recvbuf[root], deviceId, sendbuf);
  return devapis::diclResult_t::DICL_SUCCESS;
}

devapis::diclResult_t diclScatter(void* const* sendbuf, void* recvbuf,
                                  size_t count, at::ScalarType datatype,
                                  int root, int curRank, int numRanks,
                                  diclComm_t comm, deviceStream_t stream) {
  if (curRank != root) {
    DIPU_CALL_DICLAPIS(diclRecv(recvbuf, count, datatype, root, comm, stream));
    return devapis::diclResult_t::DICL_SUCCESS;
  }

  for (const auto dstRank : c10::irange(numRanks)) {
    if (dstRank == root) {
      continue;
    }
    DIPU_CALL_DICLAPIS(
        diclSend(sendbuf[dstRank], count, datatype, dstRank, comm, stream));
  }

  auto deviceId = static_cast<devapis::deviceId_t>(curRank);
  devapis::memCopyD2DAsync(stream, count * c10::elementSize(datatype), deviceId,
                           recvbuf, deviceId, sendbuf[root]);
  return devapis::diclResult_t::DICL_SUCCESS;
}

devapis::diclResult_t diclReduce(const void* sendbuff, void* recvbuff,
                                 size_t count, at::ScalarType datatype,
                                 const devapis::ReduceOp& reduceOp, int root,
                                 diclComm_t comm, deviceStream_t stream) {
  return devapis::diclReduce(sendbuff, recvbuff, count, datatype, reduceOp,
                             root, comm, stream);
}

devapis::diclResult_t diclReduceScatter(
    void* sendbuff, void* recvbuff, size_t recvCount, at::ScalarType datatype,
    const devapis::ReduceOp& op, diclComm_t comm, deviceStream_t stream) {
  return devapis::diclReduceScatter(sendbuff, recvbuff, recvCount, datatype, op,
                                    comm, stream);
}

devapis::diclResult_t diclSend(void* sendbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  return devapis::diclSend(sendbuff, count, datatype, peer, comm, stream);
}

devapis::diclResult_t diclRecv(void* recvbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  return devapis::diclRecv(recvbuff, count, datatype, peer, comm, stream);
}

devapis::diclResult_t diclGetCommName(std::string& commName, diclComm_t comm) {
  if (devapis::diclGetCommName) {
    return devapis::diclGetCommName(commName, comm);
  }
  TORCH_CHECK(false, "device not implement diclGetCommName");
}

}  // namespace devproxy
}  // namespace dipu
