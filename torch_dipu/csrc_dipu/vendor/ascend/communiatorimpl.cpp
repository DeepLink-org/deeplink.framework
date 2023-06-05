#include <acl/acl.h>
#include <cstring>
#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

// HCCL ReduceOp mapping
std::map<c10d::ReduceOp, HcclReduceOp> hcclOp = {
  {ReduceOp::MIN, HCCL_REDUCE_MIN},
  {ReduceOp::MAX, HCCL_REDUCE_MAX},
  {ReduceOp::SUM, HCCL_REDUCE_SUM},
  {ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
};

// HCCL DataType mapping
std::map<at::ScalarType, HcclDataType> hcclDataType = {
  {at::kByte, HCCL_DATA_TYPE_UINT8},
  {at::kChar, HCCL_DATA_TYPE_INT8},
  {at::kShort, HCCL_DATA_TYPE_INT16},
  {at::kInt, HCCL_DATA_TYPE_INT32},
  {at::kLong, HCCL_DATA_TYPE_INT64},
  {at::kHalf, HCCL_DATA_TYPE_FP16},
  {at::kFloat, HCCL_DATA_TYPE_FP32},
  {at::kDouble, HCCL_DATA_TYPE_FP64},
  {at::kBool, HCCL_DATA_TYPE_UINT8},
};

#define HCCL_ASSERT(cmd)                                            \
  do {                                                              \
    HcclResult error = cmd;                                         \
    if (error != HCCL_SUCCESS) {                                    \
      std::string err = "HCCL error in: " + std::string(__FILE__) + \
          ":" + std::to_string(__LINE__) + ".\n" +                  \
          "And see details in Ascend logs.\n" +                     \
          aclGetRecentErrMsg();                                     \
      throw std::runtime_error(err);                                \
    }                                                               \
  } while (0)

// std::string getHcclDataTypeSerialString(HcclDataType type){
//   const auto &iter = kHcclDataTypeToStringMap.find(type);
//   if (iter != kHcclDataTypeToStringMap.end()){
//     return iter->second;
//   } else {
//     TORCH_WARN_ONCE("Can not serialize undefined hccl data type.");
//     return "";
//   }
// }

// // AllGather & Broadcast support all data type, no need do more check.
// void checkSupportedDataTypeOfAllReduce(HcclDataType type) {
//   static std::set <HcclDataType> allReduceSupportedDataTypes = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16,
//                                                                 HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_FP16,
//                                                                 HCCL_DATA_TYPE_FP32};
//   TORCH_CHECK(allReduceSupportedDataTypes.count(type) != 0,
//               "HCCL AllReduce & Reduce: Unsupported data type ",
//               getHcclDataTypeSerialString(type));
// }


const int DICL_UNIQUE_ID_BYTES_SIZE = HCCL_ROOT_INFO_BYTES;

DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  throw std::runtime_error("ascend Not implement diclGetCommAsyncError");
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  HCCL_ASSERT(HcclGetRootInfo(uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId,
                                        int rank, int localDeviceId) {
  HCCL_ASSERT(HcclCommInitRootInfo(nranks, &uniqueId, rank, comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
  HCCL_ASSERT(HcclCommDestroy(comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                                    at::ScalarType datatype, const ReduceOp& reduceOp,
                                    diclComm_t comm, deviceStream_t stream) {
  HCCL_ASSERT(HcclAllReduce(const_cast<void *>(sendbuff), recvbuff, count, hcclDataType[datatype], hcclOp[reduceOp], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void *sendBuf, void *recvBuf, size_t count,
                                    at::ScalarType datatype, diclComm_t comm, deviceStream_t stream) {
  HCCL_ASSERT(HcclAllGather(const_cast<void *>(sendBuf), recvBuf, count, hcclDataType[datatype], comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream) {
  throw std::runtime_error("ascend Not implement diclReduce");
}

DIPU_API diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, at::ScalarType dataType, 
                                  const ReduceOp& op, diclComm_t comm, deviceStream_t stream) {
  throw std::runtime_error("ascend Not implement diclReduceScatter");
}

DIPU_API diclResult_t diclSend(void* sendbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream){
  throw std::runtime_error("ascend Not implement diclSend");
}

DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream) {
  throw std::runtime_error("ascend Not implement diclRecv");
}

  DIPU_API diclResult_t diclBroadcast(const void *sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                              int root, diclComm_t comm, deviceStream_t stream) {
    throw std::runtime_error("ascend Not implement diclBroadcast");
  }

} // end namespace devapis
} // end namespace dipu