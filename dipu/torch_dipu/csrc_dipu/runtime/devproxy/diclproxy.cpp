// Copyright (c) 2023, DeepLink.

#include "diclproxy.h"

#include <vector>

#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>

#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/runtime/device/diclapis.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include <csrc_dipu/vendor/vendorapi.h>

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

devapis::diclResult_t diclGather(const void* sendbuf, void* const* recvbuf,
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

devapis::diclResult_t diclScatter(const void* const* sendbuf, void* recvbuf,
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

devapis::diclResult_t diclAllToAllEqualSplit(
    const void* sendBuf, void* recvBuf, size_t count, at::ScalarType dataType,
    diclComm_t comm, deviceStream_t stream, int currRank, int commSize) {
  if (devapis::diclAllToAllEqualSplit) {
    return devapis::diclAllToAllEqualSplit(sendBuf, recvBuf, count, dataType,
                                           comm, stream);
  }

  // TODO(jfxu-st): For CUDA, use NCCL Group Calls for higher performance
  // Ref:
  // https://github.com/pytorch/pytorch/blob/f2d7f235a684c593f5a1ff2ca0b47b47274bfe85/torch/csrc/cuda/nccl.cpp#L828-L838
  // Ref:
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-all
  TORCH_WARN_ONCE(
      "devapis::diclAllToAllEqualSplit is not implemented, so a fallback "
      "implementation based on devproxy::diclScatter will be used")
  const size_t numBytesPerRank = count * c10::elementSize(dataType);
  std::vector<const void*> sendBuf2d(commSize);
  for (const auto scatterRootRank : c10::irange(commSize)) {
    sendBuf2d[scatterRootRank] = reinterpret_cast<const char*>(sendBuf) +
                                 scatterRootRank * numBytesPerRank;
  }
  for (const auto peer : c10::irange(commSize)) {
    diclScatter(sendBuf2d.data(),
                reinterpret_cast<char*>(recvBuf) + peer * numBytesPerRank,
                count, dataType, peer, currRank, commSize, comm, stream);
  }
  return devapis::DICL_SUCCESS;
}

DIPU_API devapis::diclResult_t diclAllToAllUnequalSplit(
    const void* sendBuf, const size_t* sendCounts,
    const size_t* sendDisplacements, void* recvBuf, const size_t* recvCounts,
    const size_t* recvDisplacements, at::ScalarType dataType, diclComm_t comm,
    deviceStream_t stream, int currRank, int commSize) {
  if (devapis::diclAllToAllUnequalSplit) {
    return devapis::diclAllToAllUnequalSplit(
        sendBuf, sendCounts, sendDisplacements, recvBuf, recvCounts,
        recvDisplacements, dataType, comm, stream);
  }

  // TODO(jfxu-st): For CUDA, use NCCL Group Calls for higher performance
  // Ref:
  // https://github.com/pytorch/pytorch/blob/f2d7f235a684c593f5a1ff2ca0b47b47274bfe85/torch/csrc/cuda/nccl.cpp#L871-L893

  TORCH_WARN_ONCE(
      "devapis::diclAllToAllUnequalSplit is not implemented, so a fallback "
      "implementation based on devproxy::diclSend and devproxy::diclRecv will "
      "be used")

  size_t elementSize = c10::elementSize(dataType);
  for (const auto scatterRootRank : c10::irange(commSize)) {
    if (currRank != scatterRootRank) {
      DIPU_CALL_DICLAPIS(
          diclRecv(reinterpret_cast<char*>(recvBuf) +
                       recvDisplacements[scatterRootRank] * elementSize,
                   recvCounts[scatterRootRank], dataType, scatterRootRank, comm,
                   stream));
      continue;
    }

    for (const auto dstRank : c10::irange(commSize)) {
      if (dstRank == scatterRootRank) {
        continue;
      }
      DIPU_CALL_DICLAPIS(diclSend(reinterpret_cast<const char*>(sendBuf) +
                                      sendDisplacements[dstRank] * elementSize,
                                  sendCounts[dstRank], dataType, dstRank, comm,
                                  stream));
    }

    auto deviceId = static_cast<devapis::deviceId_t>(currRank);
    devproxy::memCopyD2DAsync(stream, sendCounts[currRank] * elementSize,
                              deviceId,
                              reinterpret_cast<char*>(recvBuf) +
                                  recvDisplacements[currRank] * elementSize,
                              deviceId,
                              reinterpret_cast<const char*>(sendBuf) +
                                  sendDisplacements[currRank] * elementSize);
  }
  return devapis::DICL_SUCCESS;
}

devapis::diclResult_t diclSend(const void* sendbuff, size_t count,
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
