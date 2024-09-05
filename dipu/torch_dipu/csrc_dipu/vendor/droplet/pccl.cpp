#include "pccl.h"
#include <cstddef>
#include "pcclcommon.h"
#include <c10/core/ScalarType.h>

#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>


namespace {
template <const char* workspaceApi, typename... Args>
void callPcclImpl(Args... args) {
    static const auto workspaceSizeFuncAddr = getCommPcclFuncAddr(workspaceApi);
    using WorkspaceSizeFunc = int (*)(Args...);
    static WorkspaceSizeFunc workspaceSizeFunc = reinterpret_cast<WorkspaceSizeFunc>(workspaceSizeFuncAddr);
    auto workspaceStatus = workspaceSizeFunc(args...);
    if (workspaceStatus != pcclSuccess) {
        throw std::runtime_error(
            std::string("[") + workspaceApi + "]'s return value is not equal to PCCL_SUCCESS. pcclStatus is " + std::to_string(workspaceStatus)
        );
    }
}

#define DIPU_PCCL_IMPL(NAME, ...) \
  pcclResult_t NAME(DIPU_TYPE_PARAM(__VA_ARGS__)) { \
    static constexpr const char fstr[] = #NAME; \
    callPcclImpl<fstr>(DIPU_PARAM(__VA_ARGS__)); \
    return pcclSuccess; \
  } \
  static pcclResult_t CONCAT(my__, NAME)(DIPU_TYPE_PARAM(__VA_ARGS__)); \
  static const int CONCAT(n_, NAME) = []() { \
    fn[#NAME] = reinterpret_cast<void*>(CONCAT(my__, NAME)); \
    return 0; \
  }(); \
  pcclResult_t CONCAT(my__, NAME)(DIPU_TYPE_PARAM(__VA_ARGS__))

std::map<std::string, void*> fn;


static const std::map<pcclDataType_t, at::ScalarType> toScalarType = {
    {pcclInt8, at::kChar},
    {pcclUint8, at::kByte},
    {pcclFloat, at::kFloat},
    // TODO: PCCL not support double now
    // {pcclDouble, at::kDouble},
    {pcclInt32, at::kInt},
    {pcclInt64, at::kLong},
    {pcclHalf, at::kHalf},
    {pcclUint8, at::kBool},
    {pcclBfloat16, at::kBFloat16},
};

at::ScalarType PcclDataTypeToScalarType(pcclDataType_t pccl_data_type) {
    auto p = toScalarType.find(pccl_data_type);
    if (p == toScalarType.end()) {
        throw std::runtime_error("Not supported pcclDataType_t: " + std::to_string(pccl_data_type));
    }
    return p->second;
}

static const pcclComm_t kMagicComm = reinterpret_cast<pcclComm_t>(0x5043434C);

void checkCommOrThrow(pcclComm_t comm) {
  if (comm == nullptr || comm!=kMagicComm) {
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

void singleDeviceMemcpy(dipu::deviceStream_t stream, void* dst, const void* src,
                        size_t nbytes) {
  auto device = dipu::devproxy::current_device();
  dipu::devproxy::memCopyD2DAsync(stream, nbytes, device, dst, device, src);
}

}  // namespace




  DIPU_PCCL_IMPL(pcclGetUniqueId, (pcclUniqueId*, uniqueId)) {
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommInitRank, (pcclComm_t*, comm), (int, ndev), (pcclUniqueId, commIdI), (int, rank)) {
    checkNrankOrThrow(ndev);
    checkRankOrThrow(rank);
    DIPU_LOGW(
        "PCCL is not enabled. DIPU will simulate single GPU "
        "communication using memcpy.");
        *comm = kMagicComm;
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommDestroy, (pcclComm_t, comm)) {
    checkCommOrThrow(comm);
    // destroyDiclComm(comm);   
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclCommGetAsyncError, (pcclComm_t, comm), (pcclResult_t*, asyncError)) {
    checkCommOrThrow(comm);
    return pcclSuccess;
  }

const char* pcclGetErrorString(pcclResult_t result){
  // Not Fallback
  static const char* apiName = "pcclGetErrorString";
  static const auto funcptr = getCommPcclFuncAddr(apiName);
  using func = const char*(*)(pcclResult_t);
  return reinterpret_cast<func>(funcptr)(result);
}

const char* pcclGetLastError(pcclComm_t comm){
  // Not Fallback
  static const char* apiName = "pcclGetLastError";
  static const auto funcptr = getCommPcclFuncAddr(apiName);
  using func = const char*(*)(pcclComm_t);
  return reinterpret_cast<func>(funcptr)(comm);
}


  DIPU_PCCL_IMPL(pcclReduce, (const void*, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (pcclRedOp_t, op), (int, root), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    checkRankOrThrow(root);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                         count * at::elementSize(PcclDataTypeToScalarType(datatype)));
    }
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclAllReduce, (const void*, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (pcclRedOp_t, op), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                         count * at::elementSize(PcclDataTypeToScalarType(datatype)));
    }
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclReduceScatter, (const void*, sendbuff), (void*, recvbuff), (size_t, recvcount), (pcclDataType_t, datatype), (pcclRedOp_t, op), (pcclComm_t, comm), (tangStream_t, stream)) {
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                   recvcount * at::elementSize(PcclDataTypeToScalarType(datatype)));
    }
    return pcclSuccess;
  }

  DIPU_PCCL_IMPL(pcclBroadcast, (const void *, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (int, root), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                       count * at::elementSize(PcclDataTypeToScalarType(datatype)));
    }
    return pcclSuccess;
  }
  DIPU_PCCL_IMPL(pcclAllGather, (const void*, sendbuff), (void*, recvbuff), (size_t, count), (pcclDataType_t, datatype), (pcclComm_t, comm), (tangStream_t, stream)) {
    checkCommOrThrow(comm);
    if(sendbuff != recvbuff){
      singleDeviceMemcpy(stream, recvbuff, sendbuff,
                         count * at::elementSize(PcclDataTypeToScalarType(datatype)));
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

