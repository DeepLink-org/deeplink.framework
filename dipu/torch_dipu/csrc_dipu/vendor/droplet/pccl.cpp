/**
 * pccl.cpp
 *
 * Description:
 * This file implements the dynamic loading and invocation of PCCL APIs required
 * by DICL. If the pccllib.so library is not found, a log message will be
 * printed, and a Fallback API will be executed.
 *
 * Notes:
 * - We have copied the PCCL header file. If the PCCL header file are updated,
 * please correspondingly update them here.
 */
#include "pccl.h"

#include <cstddef>
#include <stdexcept>

#include "pcclcommon.h"

#include <c10/core/ScalarType.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include "csrc_dipu/runtime/device/basedef.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"
#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace {

#define DIPU_PCCL_IMPL(NAME, RETURN, ...)                                     \
  RETURN NAME(DIPU_TYPE_PARAM(__VA_ARGS__)) {                                 \
    static constexpr const char fstr[] = #NAME;                               \
    return callPcclImpl<fstr, RETURN>(DIPU_PARAM(__VA_ARGS__));               \
  }                                                                           \
  static RETURN CONCAT(my__, NAME)(DIPU_TYPE_PARAM(__VA_ARGS__));             \
  static const int CONCAT(n_, NAME) = []() {                                  \
    g_pccl_function_map[#NAME] = reinterpret_cast<void*>(CONCAT(my__, NAME)); \
    return 0;                                                                 \
  }();                                                                        \
  RETURN CONCAT(my__, NAME)(DIPU_TYPE_PARAM(__VA_ARGS__))

#define DIPU_PCCL_COMM_IMPL(NAME, ...) \
  DIPU_PCCL_IMPL(NAME, pcclResult_t, __VA_ARGS__)
#define DIPU_PCCL_ERROR_IMPL(NAME, ...) \
  DIPU_PCCL_IMPL(NAME, const char*, __VA_ARGS__)

std::map<std::string, void*> g_pccl_function_map;

template <const char* PcclFuncName, typename ReturnType, typename... Args>
ReturnType callPcclImpl(Args... args) {
  static const auto functionAddress = getCommPcclFuncAddr(PcclFuncName);
  using PcclFuncType = ReturnType (*)(Args...);
  static PcclFuncType pcclFunc = reinterpret_cast<PcclFuncType>(
      functionAddress != nullptr ? functionAddress
                                 : g_pccl_function_map[PcclFuncName]);
  auto pcclCallReturn = pcclFunc(args...);
  return pcclCallReturn;
}

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
  TORCH_CHECK(p != toScalarType.end(), "Not supported pcclDataType_t: " +
                                           std::to_string(pccl_data_type));
  return p->second;
}

static const pcclComm_t kMagicComm = reinterpret_cast<pcclComm_t>(0x5043434C);

void checkCommOrThrow(pcclComm_t comm) {
  TORCH_CHECK(comm != nullptr && comm == kMagicComm, "Invalid comm.");
}

[[noreturn]] void throwNotSupportedError() {
  TORCH_CHECK(
      false, "PCCL is not enabled. DIPU only allows single GPU communication.");
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
  if (dst != src) {
    auto device = dipu::devproxy::current_device();
    dipu::devproxy::memCopyD2DAsync(stream, nbytes, device, dst, device, src);
  }
}

}  // namespace

DIPU_PCCL_COMM_IMPL(pcclGetUniqueId, (pcclUniqueId*, uniqueId)) {
  return pcclSuccess;
}

DIPU_PCCL_COMM_IMPL(pcclCommInitRank, (pcclComm_t*, comm), (int, ndev),
                    (pcclUniqueId, commIdI), (int, rank)) {
  checkNrankOrThrow(ndev);
  checkRankOrThrow(rank);
  DIPU_LOGW(
      "PCCL is not enabled. DIPU will simulate single GPU "
      "communication using memcpy.");
  *comm = kMagicComm;
  return pcclSuccess;
}

DIPU_PCCL_COMM_IMPL(pcclCommDestroy, (pcclComm_t, comm)) {
  checkCommOrThrow(comm);
  // destroyDiclComm(comm);
  return pcclSuccess;
}

DIPU_PCCL_COMM_IMPL(pcclCommGetAsyncError, (pcclComm_t, comm),
                    (pcclResult_t*, asyncError)) {
  checkCommOrThrow(comm);
  return pcclSuccess;
}

DIPU_PCCL_ERROR_IMPL(pcclGetErrorString, (pcclResult_t, result)) {
  TORCH_CHECK(false, "Fallback pccl impl should not call pcclGetErrorString");
}

DIPU_PCCL_ERROR_IMPL(pcclGetLastError, (pcclComm_t, comm)) {
  TORCH_CHECK(false, "Fallback pccl impl should not call pcclGetLastError");
}

DIPU_PCCL_COMM_IMPL(pcclReduce, (const void*, sendbuff), (void*, recvbuff),
                    (size_t, count), (pcclDataType_t, datatype),
                    (pcclRedOp_t, op), (int, root), (pcclComm_t, comm),
                    (tangStream_t, stream)) {
  checkCommOrThrow(comm);
  checkRankOrThrow(root);
  singleDeviceMemcpy(
      stream, recvbuff, sendbuff,
      count * at::elementSize(PcclDataTypeToScalarType(datatype)));
  return pcclSuccess;
}

DIPU_PCCL_COMM_IMPL(pcclAllReduce, (const void*, sendbuff), (void*, recvbuff),
                    (size_t, count), (pcclDataType_t, datatype),
                    (pcclRedOp_t, op), (pcclComm_t, comm),
                    (tangStream_t, stream)) {
  checkCommOrThrow(comm);
  singleDeviceMemcpy(
      stream, recvbuff, sendbuff,
      count * at::elementSize(PcclDataTypeToScalarType(datatype)));
  return pcclSuccess;
}

DIPU_PCCL_COMM_IMPL(pcclReduceScatter, (const void*, sendbuff),
                    (void*, recvbuff), (size_t, recvcount),
                    (pcclDataType_t, datatype), (pcclRedOp_t, op),
                    (pcclComm_t, comm), (tangStream_t, stream)) {
  singleDeviceMemcpy(
      stream, recvbuff, sendbuff,
      recvcount * at::elementSize(PcclDataTypeToScalarType(datatype)));
  return pcclSuccess;
}

DIPU_PCCL_COMM_IMPL(pcclBroadcast, (const void*, sendbuff), (void*, recvbuff),
                    (size_t, count), (pcclDataType_t, datatype), (int, root),
                    (pcclComm_t, comm), (tangStream_t, stream)) {
  checkCommOrThrow(comm);
  singleDeviceMemcpy(
      stream, recvbuff, sendbuff,
      count * at::elementSize(PcclDataTypeToScalarType(datatype)));
  return pcclSuccess;
}
DIPU_PCCL_COMM_IMPL(pcclAllGather, (const void*, sendbuff), (void*, recvbuff),
                    (size_t, count), (pcclDataType_t, datatype),
                    (pcclComm_t, comm), (tangStream_t, stream)) {
  checkCommOrThrow(comm);
  singleDeviceMemcpy(
      stream, recvbuff, sendbuff,
      count * at::elementSize(PcclDataTypeToScalarType(datatype)));
  return pcclSuccess;
}
DIPU_PCCL_COMM_IMPL(pcclSend, (const void*, sendbuff), (size_t, count),
                    (pcclDataType_t, datatype), (int, peer), (pcclComm_t, comm),
                    (tangStream_t, stream)) {
  throwNotSupportedError();
  return pcclInvalidUsage;
}
DIPU_PCCL_COMM_IMPL(pcclRecv, (void*, recvbuff), (size_t, count),
                    (pcclDataType_t, datatype), (int, peer), (pcclComm_t, comm),
                    (tangStream_t, stream)) {
  throwNotSupportedError();
  return pcclInvalidUsage;
}
