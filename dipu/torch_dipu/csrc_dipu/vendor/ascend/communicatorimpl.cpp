
#include "basecommimpl.hpp"

#define HCCL_THROW(cmd)                                           \
  do {                                                            \
    TRACK_FUN_CALL(HCCL, #cmd);                                   \
    TORCH_CHECK(cmd == HCCL_SUCCESS,                              \
                "HCCL error in: " + std::string(__FILE__) + ":" + \
                    std::to_string(__LINE__) + ".\n" +            \
                    "And see details in Ascend logs.\n" +         \
                    aclGetRecentErrMsg());                        \
  } while (0)

namespace dipu {
namespace devapis {

constexpr const int MAX_COMM_NAME_LENGTH = 128;

// HCCL ReduceOp mapping
static std::map<c10d::ReduceOp, HcclReduceOp> hcclOp = {
    {ReduceOp::MIN, HCCL_REDUCE_MIN},
    {ReduceOp::MAX, HCCL_REDUCE_MAX},
    {ReduceOp::SUM, HCCL_REDUCE_SUM},
    {ReduceOp::PRODUCT, HCCL_REDUCE_PROD},
};

// HCCL DataType mapping
static constexpr std::array<std::pair<at::ScalarType, HcclDataType>, 10>
    hcclDataTypes{{
        {at::kByte, HCCL_DATA_TYPE_UINT8},
        {at::kChar, HCCL_DATA_TYPE_INT8},
        {at::kShort, HCCL_DATA_TYPE_INT16},
        {at::kInt, HCCL_DATA_TYPE_INT32},
        {at::kLong, HCCL_DATA_TYPE_INT64},
        {at::kHalf, HCCL_DATA_TYPE_FP16},
        {at::kFloat, HCCL_DATA_TYPE_FP32},
        {at::kDouble, HCCL_DATA_TYPE_FP64},
        {at::kBool, HCCL_DATA_TYPE_UINT8},
        {at::kBFloat16, HCCL_DATA_TYPE_BFP16},
    }};

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

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
                                       commUniqueId uniqueId, int rank,
                                       int localDeviceId) {
  HCCL_THROW(HcclCommInitRootInfo(nranks, &uniqueId, rank, comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
  HCCL_THROW(HcclCommDestroy(comm));
  return DICL_SUCCESS;
}

std::string getHcclDataTypeSerialString(HcclDataType type) {
  switch (type) {
    case HCCL_DATA_TYPE_UINT8:
      return "at::kByte/at::kBool";
    case HCCL_DATA_TYPE_INT8:
      return "at::kChar";
    case HCCL_DATA_TYPE_INT16:
      return "at::kShort";
    case HCCL_DATA_TYPE_INT32:
      return "at::kInt";
    case HCCL_DATA_TYPE_INT64:
      return "at::kLong";
    case HCCL_DATA_TYPE_FP16:
      return "at::kHalf";
    case HCCL_DATA_TYPE_FP32:
      return "at::kFloat";
    case HCCL_DATA_TYPE_FP64:
      return "at::kDouble";
    case HCCL_DATA_TYPE_BFP16:
      return "at::kBFloat16";
    default:
      TORCH_WARN_ONCE("Can not serialize undefined hccl data type.");
  }
  return "";
}

void checkSupportedDataTypeOfAllReduce(HcclDataType type) {
  static const std::unordered_set<HcclDataType> allReduceSupportedDataTypes = {
      HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32,
      HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32,  HCCL_DATA_TYPE_BFP16,
      HCCL_DATA_TYPE_INT64};
  TORCH_CHECK(allReduceSupportedDataTypes.count(type) != 0,
              "HCCL AllReduce & Reduce: Unsupported data type ",
              getHcclDataTypeSerialString(type));
}

DIPU_API diclResult_t diclAllReduce(const void* sendBuff, void* recvBuff,
                                    size_t count, at::ScalarType dataType,
                                    const ReduceOp& reduceOp, diclComm_t comm,
                                    deviceStream_t stream) {
  // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC1alpha003/apiref/hcclapiref/hcclcpp_07_0014.html
  checkSupportedDataTypeOfAllReduce(getHcclDataType(dataType));
  HCCL_THROW(HcclAllReduce(const_cast<void*>(sendBuff), recvBuff, count,
                           getHcclDataType(dataType), hcclOp[reduceOp], comm,
                           stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void* sendBuf, void* recvBuf,
                                    size_t count, at::ScalarType dataType,
                                    diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(HcclAllGather(const_cast<void*>(sendBuf), recvBuf, count,
                           getHcclDataType(dataType), comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendBuf, void* recvBuf,
                                 size_t count, at::ScalarType dataType,
                                 const ReduceOp& reduceOp, int root,
                                 diclComm_t comm, deviceStream_t stream) {
  checkSupportedDataTypeOfAllReduce(getHcclDataType(dataType));
  HCCL_THROW(HcclReduce(const_cast<void*>(sendBuf), recvBuf, count,
                        getHcclDataType(dataType), hcclOp[reduceOp], root, comm,
                        stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduceScatter(void* sendBuf, void* recvBuf,
                                        size_t recvCount,
                                        at::ScalarType dataType,
                                        const ReduceOp& op, diclComm_t comm,
                                        deviceStream_t stream) {
  HCCL_THROW(HcclReduceScatter(sendBuf, recvBuf, recvCount,
                               getHcclDataType(dataType), hcclOp[op], comm,
                               stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclSend(void* sendBuf, size_t count,
                               at::ScalarType dataType, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(
      HcclSend(sendBuf, count, getHcclDataType(dataType), peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRecv(void* recvBuf, size_t count,
                               at::ScalarType dataType, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  HCCL_THROW(
      HcclRecv(recvBuf, count, getHcclDataType(dataType), peer, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclBroadcast(const void* sendBuf, void* recvBuf,
                                    size_t count, at::ScalarType dataType,
                                    int root, diclComm_t comm,
                                    deviceStream_t stream) {
  HCCL_THROW(HcclBroadcast(const_cast<void*>(sendBuf), count,
                           getHcclDataType(dataType), root, comm, stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclGetCommName(std::string& commName, diclComm_t comm) {
  std::array<char, MAX_COMM_NAME_LENGTH> commName_{};
  HCCL_THROW(HcclGetCommName(comm, commName_.data()));
  commName = std::string{commName_.data()};
  return DICL_SUCCESS;
}

}  // end namespace devapis
}  // end namespace dipu
