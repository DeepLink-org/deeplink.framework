#include <cstring>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

// NCCL op mapping
static const std::map<ReduceOp::RedOpType, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin}, {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum}, {ReduceOp::PRODUCT, ncclProd},
#ifdef NCCL_HAS_AVG
    {ReduceOp::AVG, ncclAvg},
#endif
};

// NCCL type typing
static const std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},         {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},       {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},         {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},         {at::kBool, ncclUint8},
#if HAS_NCCL_BF16_DATATYPE
    {at::kBFloat16, ncclBfloat16},
#endif
};

// Macro to print and abort on a non-successful NCCL return value.
#define NCCL_THROW(cmd)                                                 \
  do {                                                                  \
    ncclResult_t result = cmd;                                          \
    if (result != ncclSuccess) {                                        \
      std::string err = ncclGetErrorString(result);                     \
      fprintf(stderr, "NCCL error in: %s:%d, %s\n", __FILE__, __LINE__, \
              err.c_str());                                             \
      TORCH_CHECK(false, err);                                          \
    }                                                                   \
  } while (0)

const int DICL_UNIQUE_ID_BYTES_SIZE = NCCL_UNIQUE_ID_BYTES;

DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  ncclResult_t ncclAsyncErr_;
  NCCL_THROW(ncclCommGetAsyncError(comm, &ncclAsyncErr_));
  if (ncclAsyncErr_ != ncclSuccess) {
    return DICL_SUCCESS;
  }
  return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  NCCL_THROW(ncclGetUniqueId(uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
                                       commUniqueId uniqueId, int rank,
                                       int localDeviceId) {
  NCCL_THROW(ncclCommInitRank(comm, nranks, uniqueId, rank));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(ncclComm_t comm) {
  NCCL_THROW(ncclCommDestroy(comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllReduce(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    const ReduceOp& reduceOp, diclComm_t comm,
                                    deviceStream_t stream) {
  NCCL_THROW(ncclAllReduce(sendbuff, recvbuff, count, ncclDataType.at(datatype),
                           ncclOp.at(reduceOp), comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclBroadcast(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    int root, diclComm_t comm,
                                    deviceStream_t stream) {
  NCCL_THROW(ncclBroadcast(sendbuff, recvbuff, count, ncclDataType.at(datatype),
                           root, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void* sendBuf, void* recvBuf,
                                    size_t sendCount, at::ScalarType datatype,
                                    diclComm_t comm, deviceStream_t stream) {
  NCCL_THROW(ncclAllGather(sendBuf, recvBuf, sendCount,
                           ncclDataType.at(datatype), comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff,
                                 size_t count, at::ScalarType datatype,
                                 const ReduceOp& reduceOp, int root,
                                 diclComm_t comm, deviceStream_t stream) {
  NCCL_THROW(ncclReduce(sendbuff, recvbuff, count, ncclDataType.at(datatype),
                        ncclOp.at(reduceOp), root, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduceScatter(
    void* sendBuf, void* recvBuf, size_t recvCount, at::ScalarType datatype,
    const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
  NCCL_THROW(ncclReduceScatter(sendBuf, recvBuf, recvCount,
                               ncclDataType.at(datatype), ncclOp.at(reduceOp),
                               comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclSend(void* sendbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  NCCL_THROW(
      ncclSend(sendbuff, count, ncclDataType.at(datatype), peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  NCCL_THROW(
      ncclRecv(recvbuff, count, ncclDataType.at(datatype), peer, comm, stream));
  return DICL_SUCCESS;
}

}  // end namespace devapis
}  // end namespace dipu
