#include "pccl.h"
#include <cstddef>
#include "pcclcommon.h"
#include <c10/core/ScalarType.h>

#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>


template <const char* workspaceApi, typename... Args>
int callPcclImpl(Args... args) {
    static const auto workspaceSizeFuncAddr = getOpApiFuncAddr(workspaceApi);
    using WorkspaceSizeFunc = int (*)(Args...);
    static WorkspaceSizeFunc workspaceSizeFunc = reinterpret_cast<WorkspaceSizeFunc>(workspaceSizeFuncAddr);
    auto workspaceStatus = workspaceSizeFunc(args...);
    if (workspaceStatus != pcclSuccess) {
        throw std::runtime_error(
            std::string("[") + workspaceApi + "]'s return value is not equal to PCCL_SUCCESS. pcclStatus is " + std::to_string(workspaceStatus)
        );
    }
    return workspaceStatus;
}

#define DIPU_PCCL_IMPL(NAME, ...) \
  pcclResult_t NAME(TYPE_PARAM(__VA_ARGS__)) { \
    static constexpr const char fstr[] = #NAME; \
    callPcclImpl<fstr>(PARAM(__VA_ARGS__)); \
  } \
  static pcclResult_t CONCAT(my__, NAME)(TYPE_PARAM(__VA_ARGS__)); \
  static const int CONCAT(n_, NAME) = []() { \
    fn[#NAME] = reinterpret_cast<void*>(CONCAT(my__, NAME)); \
    return 0; \
  }(); \
  pcclResult_t CONCAT(my__, NAME)(TYPE_PARAM(__VA_ARGS__))

std::map<std::string, void*> fn;

namespace {

using diclCommValue_t = std::remove_pointer_t<pcclComm_t>;
constexpr diclCommValue_t kMagicComm = 0x5043434C;  // "PCCL"

pcclComm_t createDiclComm() { return new diclCommValue_t(kMagicComm); }

void destroyDiclComm(pcclComm_t comm) { delete comm; }

void checkCommOrThrow(pcclComm_t comm) {
  if (comm == nullptr || *comm != kMagicComm) {
    throw std::runtime_error("Invalid comm.");
  }
}

[[noreturn]] void throwNotSupportedError() {
  throw std::runtime_error(
      "PCCL is not enabled. DIPU only allows single GPU communication.");
}

void checkNrankOrThrow(int nranks) {
  if (nranks != 1) {
    throwNotSupportedError();
  }
}

void checkRankOrThrow(int rank) {
  if (rank != 0) {
    throwNotSupportedError();
  }
}

void singleDeviceMemcpy(deviceStream_t stream, void* dst, const void* src,
                        size_t nbytes) {
  auto device = devproxy::current_device();
  devproxy::memCopyD2DAsync(stream, nbytes, device, dst, device, src);
}

}  // namespace


void checkCommOrThrow(pcclComm_t comm) {
  if (comm == nullptr || *comm != kMagicComm) {
    throw std::runtime_error("Invalid comm.");
  }
}


  DIPU_PCCL_IMPL(pcclGetUniqueId, (pcclUniqueId*, uniqueId)) {
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommInitRank, (pcclComm_t*, comm), (int, ndev), (pcclUniqueId, commIdI), (int, rank)) {
    checkNrankOrThrow(ndev);
    checkRankOrThrow(rank);
    DIPU_LOGW(
        "PCCL is not enabled. DIPU will simulate single GPU "
        "communication using memcpy.");
    *comm = createDiclComm();
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommInitAll, (pcclComm_t*, comms), (int, ndev), (const int*, devlist)) {
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommDestroy, (pcclComm_t, comm)) {
    checkCommOrThrow(comm);
    destroyDiclComm(comm);   
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommAbort, (pcclComm_t, comm)) {
    return pcclSuccess;
  }
  DIPU_PCCL_IMPL(pcclCommGetAsyncError, (pcclComm_t, comm), (pcclResult_t*, asyncError)) {
    checkCommOrThrow(comm);
    return pcclSuccess;
  }
const char*  pcclGetErrorString(pcclResult_t result);
const char* pcclGetLastError(pcclComm_t comm);

  DIPU_PCCL_IMPL(pcclCommCount, (const pcclComm_t, comm), (int*, count)) {
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommCuDevice, (const pcclComm_t, comm), (int*, device)) {
    return pcclSuccess;
  }




  DIPU_PCCL_IMPL(pcclCommUserRank, (const pcclComm_t, comm), (int*, rank)) {
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclGetVersion, (int*, version)) {
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclReduce, (const void*, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (pcclRedOp_t, op), (int, root), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    checkRankOrThrow(root);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                         count * at::elementSize(datatype));
    }
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclAllReduce, (const void*, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (pcclRedOp_t, op), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                         count * at::elementSize(datatype));
    }
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclReduceScatter, (const void*, sendbuff), (void*, recvbuff), (size_t, recvcount), (pcclDataType_t, datatype), (pcclRedOp_t, op), (pcclComm_t, comm), (tangStream_t, stream)) {
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvBuf, sendBuf,
                   recvCount * at::elementSize(datatype));
    }
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclBroadcast, (const void *, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (int, root), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                       count * at::elementSize(datatype));
    }
    return pcclSuccess;
  }
  DIPU_PCCL_IMPL(pcclAllGather, (const void*, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvBuf, sendBuf,
                         count * at::elementSize(datatype));
    }
    return pcclSuccess;
  }
  DIPU_PCCL_IMPL(pcclSend, (const void*, sendbuff), (size_t, count), (pcclDataType_t, datatype), (int, peer), (pcclComm_t, comm), (tangStream_t, stream)) {
    throwNotSupportedError();
    return pcclInvalidUsage;
  }
  DIPU_PCCL_IMPL(pcclRecv, (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (int, peer), (pcclComm_t, comm), (tangStream_t, stream)) {
    throwNotSupportedError();
    return pcclInvalidUsage;
  }
  DIPU_PCCL_IMPL(pcclGroupStart, (void)) {
    return pcclSuccess;
  }
  DIPU_PCCL_IMPL(pcclGroupEnd, (void)) {
    return pcclSuccess;
  }
