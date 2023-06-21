#include <supa.h>

#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {
namespace devapis {

#define SUCCL_CALL(Expr)                                                                                               \
  {                                                                                                                    \
    succlResult_t ret = Expr;                                                                                          \
    if (ret != succlSuccess) {                                                                                         \
      std::cout << "call a succl function (" << #Expr << ") failed. return code= " << ret << std::endl;                \
      return DICL_ERR_UNDEF;                                                                                           \
    } else {                                                                                                           \
      return DICL_SUCCESS;                                                                                             \
    }                                                                                                                  \
  }

const int DICL_UNIQUE_ID_BYTES_SIZE = SUCCL_UNIQUE_ID_BYTES;

diclResult_t diclGetCommAsyncError(diclComm_t comm) {
  succlResult_t result = succlSuccess;
  SUCCL_CALL(succlCommGetAsyncError(comm, &result));
};

diclResult_t diclGetUniqueId(commUniqueId *uniqueId) { SUCCL_CALL(succlGetUniqueId(uniqueId)); }

diclResult_t diclCommInitRank(diclComm_t *comm, int nranks, commUniqueId uniqueId, int rank, int localDeviceId) {
  SUCCL_CALL(succlCommInitRank(comm, nranks, uniqueId, rank));
}

// void diclCommInitAll(diclComm_t* comms, int ndev, const int* devlist);

diclResult_t diclCommDestroy(diclComm_t comm) { SUCCL_CALL(succlCommDestroy(comm)); };

// diclResult_t diclCommFinalize(diclComm_t comm);

// diclResult_t diclCommAbort(diclComm_t comm);

static bool toSucclDataType(at::ScalarType type, succlDataType_t &out) {
  static std::map<at::ScalarType, succlDataType_t> succlDataType = {
      {at::kChar, succlInt8},
      {at::kByte, succlUint8},
      {at::kFloat, succlFloat},
      {at::kDouble, succlFloat},
      {at::kInt, succlInt32},
      {at::kLong, succlInt32},
      {at::kBool, succlUint8},
      {at::kBFloat16, succlBfloat16},
  };
  auto it = succlDataType.find(type);
  if (it == succlDataType.end()) {
    return false;
  }
  out = it->second;
  return true;
}

static bool toSucclOpType(ReduceOp type, succlRedOp_t &out) {
  static std::map<c10d::ReduceOp::RedOpType, succlRedOp_t> succlOp = {
      {c10d::ReduceOp::MIN, succlMin},
      {c10d::ReduceOp::MAX, succlMax},
      {c10d::ReduceOp::SUM, succlSum},
      {c10d::ReduceOp::PRODUCT, succlProd},
  };
  auto it = succlOp.find(type);
  if (it == succlOp.end()) {
    return false;
  }
  out = it->second;
  return true;
};

#define ConvertScalarType(x)                                                                                           \
  succlDataType_t suDataType;                                                                                          \
  if (!toSucclDataType(x, suDataType)) {                                                                               \
    return DICL_ERR_UNDEF;                                                                                             \
  }

#define ConvertOpType(x)                                                                                               \
  succlRedOp_t suOp;                                                                                                   \
  if (!toSucclOpType(x, suOp)) {                                                                                       \
    return DICL_ERR_UNDEF;                                                                                             \
  }

// SCCL op mapping
diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                           const ReduceOp &reduceOp, diclComm_t comm, deviceStream_t stream) {
  ConvertScalarType(datatype);
  ConvertOpType(reduceOp);
  SUCCL_CALL(succlAllReduce(sendbuff, recvbuff, count, suDataType, suOp, comm, stream));
}

diclResult_t diclBroadcast(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype, int root,
                           diclComm_t comm, deviceStream_t stream) {
  ConvertScalarType(datatype);
  SUCCL_CALL(succlBroadcast(sendbuff, recvbuff, count, suDataType, root, comm, stream));
}

diclResult_t diclAllGather(const void *sendBuf, void *recvBuf, size_t count, at::ScalarType datatype, diclComm_t comm,
                           deviceStream_t stream) {
  ConvertScalarType(datatype);
  SUCCL_CALL(succlAllGather(sendBuf, recvBuf, count, suDataType, comm, stream));
}

diclResult_t diclReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                        const ReduceOp &reduceOp, int root, diclComm_t comm, deviceStream_t stream) {
  ConvertScalarType(datatype);
  ConvertOpType(reduceOp);
  SUCCL_CALL(succlReduce(sendbuff, recvbuff, count, suDataType, suOp, root, comm, stream));
}

diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, at::ScalarType dataType,
                               const ReduceOp &op, diclComm_t comm, deviceStream_t stream) {
  ConvertScalarType(dataType);
  ConvertOpType(op);
  SUCCL_CALL(succlReduceScatter(sendBuf, recvBuf, recvCount, suDataType, suOp, comm, stream));
}

diclResult_t diclSend(void *sendbuff, size_t count, at::ScalarType datatype, int peer, diclComm_t comm,
                      deviceStream_t stream) {
  ConvertScalarType(datatype);
  SUCCL_CALL(succlSend(sendbuff, count, suDataType, peer, comm, stream));
}

diclResult_t diclRecv(void *recvbuff, size_t count, at::ScalarType datatype, int peer, diclComm_t comm,
                      deviceStream_t stream) {
  ConvertScalarType(datatype);
  SUCCL_CALL(succlRecv(recvbuff, count, suDataType, peer, comm, stream));
}

} // end namespace devapis
} // end namespace dipu
