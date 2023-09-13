
#include "basecommimpl.hpp"
namespace dipu {
namespace devapis {

// HCCL DataType mapping
static constexpr std::array<std::pair<at::ScalarType, HcclDataType>, 9> hcclDataTypes{
  {
    {at::kByte, HCCL_DATA_TYPE_UINT8},
    {at::kChar, HCCL_DATA_TYPE_INT8},
    {at::kShort, HCCL_DATA_TYPE_INT16},
    {at::kInt, HCCL_DATA_TYPE_INT32},
    {at::kLong, HCCL_DATA_TYPE_INT64},
    {at::kHalf, HCCL_DATA_TYPE_FP16},
    {at::kFloat, HCCL_DATA_TYPE_FP32},
    {at::kDouble, HCCL_DATA_TYPE_FP64},
    {at::kBool, HCCL_DATA_TYPE_UINT8},
  }
};

HcclDataType getHcclDataType(const at::ScalarType type) {
  static constexpr auto map =
    Map<at::ScalarType, HcclDataType, hcclDataTypes.size()>{{hcclDataTypes}};
  return map.at(type);
}

const int DICL_UNIQUE_ID_BYTES_SIZE = HCCL_ROOT_INFO_BYTES;

// TODO: not support
DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  TORCH_CHECK(false, "ascend Not implement diclGetCommAsyncError");
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  HCCL_THROW(HcclGetRootInfo(uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId,
                                        int rank, int localDeviceId) {
  HCCL_THROW(HcclCommInitRootInfo(nranks, &uniqueId, rank, comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
  HCCL_THROW(HcclCommDestroy(comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllReduce(const void *sendBuff, void *recvBuff, size_t count,
                                    at::ScalarType dataType, const ReduceOp& reduceOp,
                                    diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclAllReduce(const_cast<void *>(sendBuff), recvBuff, count,
                           getHcclDataType(dataType), hcclOp[reduceOp], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void *sendBuf, void *recvBuf, size_t count,
                                    at::ScalarType dataType, diclComm_t comm, 
                                    deviceStream_t stream) {
  HCCL_THROW(HcclAllGather(const_cast<void *>(sendBuf), recvBuf, count, 
                           getHcclDataType(dataType), comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendBuf, void* recvBuf, size_t count, 
                                 at::ScalarType dataType, const ReduceOp& reduceOp,
                                 int root, diclComm_t comm, deviceStream_t stream) {

  HCCL_THROW(HcclReduce(const_cast<void *>(sendBuf), recvBuf, count, getHcclDataType(dataType),
                        hcclOp[reduceOp], root, comm, stream));                   
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf, size_t recvCount,
                                        at::ScalarType dataType, const ReduceOp& op, 
                                        diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclReduceScatter(sendBuf, recvBuf, recvCount, getHcclDataType(dataType),
                               hcclOp[op], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclSend(void* sendBuf, size_t count, at::ScalarType dataType, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclSend(sendBuf, count, getHcclDataType(dataType), peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRecv(void* recvBuf, size_t count, at::ScalarType dataType, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclRecv(recvBuf, count, getHcclDataType(dataType), peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclBroadcast(const void *sendBuf, void* recvBuf, size_t count,
                                    at::ScalarType dataType, int root, diclComm_t comm,
                                    deviceStream_t stream) {
  HCCL_THROW(HcclBroadcast(const_cast<void *>(sendBuf), count, getHcclDataType(dataType),
                           root, comm, stream));
  return DICL_SUCCESS;
}

} // end namespace devapis
} // end namespace dipu