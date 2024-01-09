#include <bitset>
#include <bkcl.h>
#include <cstdio>
#include <cstdlib>  //注意。itoa函数要包含这个头文件
#include <cstring>
#include <stdexcept>
#include <string>

#include <c10/core/ScalarType.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

const int DICL_UNIQUE_ID_BYTES_SIZE = BKCL_UNIQUE_ID_BYTES;

#define XCCL_THROW(cmd)                                           \
  do {                                                            \
    TORCH_CHECK(cmd == BKCL_SUCCESS,                              \
                "XCCL error in: " + std::string(__FILE__) + ":" + \
                    std::to_string(__LINE__) + ".\n");            \
  } while (0)

// BKCL DataType mapping
static constexpr std::array<std::pair<at::ScalarType, BKCLDataType>, 9>
    BKCLDataTypes{{
        {at::kByte, BKCL_UINT8},
        {at::kInt, BKCL_INT32},
        {at::kLong, BKCL_INT64},
        {at::kBFloat16, BKCL_BFLOAT16},
        {at::kHalf, BKCL_FLOAT16},
        {at::kFloat, BKCL_FLOAT},
        {at::kDouble, BKCL_FLOAT64},
        {at::kBool, BKCL_UINT8},
    }};

// copy from ascend
template <typename Key, typename Value, std::size_t Size>
struct Map {
  std::array<std::pair<Key, Value>, Size> data;

  [[nodiscard]] constexpr Value at(const Key& key) const {
    const auto itr =
        std::find_if(begin(data), end(data),
                     [&key](const auto& v) { return v.first == key; });
    if (itr != end(data)) {
      return itr->second;
    } else {
      TORCH_CHECK(false, "Not Found");
    }
  }
};

// XCCL ReduceOp mapping
std::map<c10d::ReduceOp, BKCLOp> bkclOp = {
    {ReduceOp::MIN, BKCL_MIN},
    {ReduceOp::MAX, BKCL_MAX},
    {ReduceOp::SUM, BKCL_ADD},
    {ReduceOp::PRODUCT, BKCL_PRODUCT},
};

BKCLDataType getBKCLDataType(const at::ScalarType type) {
  static constexpr auto map =
      Map<at::ScalarType, BKCLDataType, BKCLDataTypes.size()>{{BKCLDataTypes}};
  return map.at(type);
}

DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
  XCCL_THROW(bkcl_get_unique_id(uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks,
                                       commUniqueId uniqueId, int rank,
                                       int localDeviceId) {
  XCCL_THROW(bkcl_init_rank(comm, rank, nranks, &uniqueId));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
  XCCL_THROW(bkcl_destroy_context(comm));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllReduce(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    const ReduceOp& reduceOp, diclComm_t comm,
                                    deviceStream_t stream) {
  XCCL_THROW(bkcl_all_reduce(comm, sendbuff, recvbuff, count,
                             getBKCLDataType(datatype), bkclOp[reduceOp],
                             stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclBroadcast(const void* sendbuff, void* recvbuff,
                                    size_t count, at::ScalarType datatype,
                                    int root, diclComm_t comm,
                                    deviceStream_t stream) {
  auto dtype = getBKCLDataType(datatype);
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void* sendBuf, void* recvBuf,
                                    size_t count, at::ScalarType datatype,
                                    diclComm_t comm, deviceStream_t stream) {
  XCCL_THROW(bkcl_all_gather(comm, sendBuf, count, recvBuf,
                             getBKCLDataType(datatype), stream));
  return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff,
                                 size_t count, at::ScalarType datatype,
                                 const ReduceOp& reduceOp, int root,
                                 diclComm_t comm, deviceStream_t stream) {
  return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclReduceScatter(
    void* sendBuf, void* recvBuf, size_t recvCount, at::ScalarType datatype,
    const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
  return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclSend(void* sendbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count,
                               at::ScalarType datatype, int peer,
                               diclComm_t comm, deviceStream_t stream) {
  return DICL_ERR_UNDEF;
}

}  // end namespace devapis

}  // end namespace dipu
