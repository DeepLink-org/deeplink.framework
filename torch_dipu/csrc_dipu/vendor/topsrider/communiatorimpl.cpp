#include "basecommimpl.hpp"

namespace dipu {

namespace devapis {

// ECCL type typing
static std::map<at::ScalarType, ecclDataType_t> eccl_data_type = {
    {at::kChar, ecclInt8},    {at::kByte, ecclUint8}, {at::kHalf, ecclHalf},
    {at::kFloat, ecclFloat},  {at::kInt, ecclInt32},  {at::kLong, ecclInt64},
    {at::kDouble, ecclDouble}};

const int DICL_UNIQUE_ID_BYTES_SIZE = ECCL_UNIQUE_ID_BYTES;

// to-do: not support
DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  // ecclResult_t result = ecclGetCommAsyncError(comm);
  // if (result != ECCL_RET_SUCCESS) {
  //   return DICL_SUCCESS;
  // } else {
  //   return DICL_ERR_UNDEF;
  // }
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  ECCL_THROW(ecclGetUniqueId(uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
                                       commUniqueId uniqueId, int rank,
                                       int localDeviceId) {
  ECCL_THROW(ecclCommInitRank(comm, nranks, uniqueId, rank));
  return DICL_SUCCESS;
}

// // DIPU_API diclResult_t diclCommInitAll(diclComm_t* comms, int ndev, const
// int* devlist);

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
  ECCL_THROW(ecclCommDestroy(comm));
  return DICL_SUCCESS;
}

// DIPU_API diclResult_t diclCommFinalize(diclComm_t comm);

// DIPU_API diclResult_t diclCommAbort(diclComm_t comm);

DIPU_API diclResult_t diclAllReduce(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    const ReduceOp& reduceOp, diclComm_t comm,
                                    deviceStream_t stream) {
  ECCL_THROW(ecclAllReduce(sendbuff, recvbuff, count, eccl_data_type[datatype],
                           eccl_op[reduceOp], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclBroadcast(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    int root, diclComm_t comm,
                                    deviceStream_t stream) {
  ECCL_THROW(ecclBroadcast(sendbuff, recvbuff, count, eccl_data_type[datatype],
                           root, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void* sendBuf, void* recvBuf,
                                    size_t count, at::ScalarType datatype,
                                    diclComm_t comm, deviceStream_t stream) {
  ECCL_THROW(ecclAllGather(sendBuf, recvBuf, count, eccl_data_type[datatype],
                           comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff,
                                 size_t count, at::ScalarType datatype,
                                 const ReduceOp& reduceOp, int root,
                                 diclComm_t comm, deviceStream_t stream) {
  ECCL_THROW(ecclReduce(sendbuff, recvbuff, count, eccl_data_type[datatype],
                        eccl_op[reduceOp], root, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduceScatter(void* sendBuf, void* recvBuf,
                                        uint64_t recvCount,
                                        at::ScalarType dataType,
                                        const ReduceOp& op, diclComm_t comm,
                                        deviceStream_t stream) {
  ECCL_THROW(ecclReduceScatter(sendBuf, recvBuf, recvCount,
                               eccl_data_type[dataType], eccl_op[op], comm,
                               stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclSend(void* sendbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  ECCL_THROW(
      ecclSend(sendbuff, count, eccl_data_type[datatype], peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  ECCL_THROW(
      ecclRecv(recvbuff, count, eccl_data_type[datatype], peer, comm, stream));
  return DICL_SUCCESS;
}

}  // end namespace devapis
}  // end namespace dipu
