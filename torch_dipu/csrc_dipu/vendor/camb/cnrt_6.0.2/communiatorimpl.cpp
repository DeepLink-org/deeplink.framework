#include "../basecommimpl.hpp"

namespace dipu {

namespace devapis {

  // CNCL type typing
  static std::map<at::ScalarType, cnclDataType_t> cncl_data_type = {
      {at::kChar, cnclInt8}, {at::kByte, cnclUint8}, {at::kHalf, cnclHalf},
      {at::kFloat, cnclFloat}, {at::kInt, cnclInt32}, {at::kLong, cnclInvalid}, 
      {at::kDouble, cnclInvalid}
  };

  static void convertTypeSize(size_t& count, at::ScalarType& datatype) {
    auto cnnltype = cncl_data_type[datatype];
    if (cnnltype == cnclDataType_t::cnclInvalid && (datatype == at::ScalarType::Long || datatype ==  at::ScalarType::Double)) {
      datatype = at::kByte;
      count = count * sizeof(long);
    }
  }

  const int DICL_UNIQUE_ID_BYTES_SIZE = CNCL_CLIQUE_ID_BYTES_SIZE;

  DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
    cnclResult_t result = cnclGetCommAsyncError(comm);
    if (result != CNCL_RET_SUCCESS) {
      return DICL_SUCCESS;
    } else {
      return DICL_ERR_UNDEF;
    }
  }

  DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
    CNCL_THROW(cnclGetCliqueId(uniqueId));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId,
                                          int rank, int localDeviceId) {
    CNCL_THROW(cnclInitComms(comm, 1, &localDeviceId, &rank, nranks, &uniqueId));
    return DICL_SUCCESS;
  }

  // // DIPU_API diclResult_t diclCommInitAll(diclComm_t* comms, int ndev, const int* devlist);

  DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
    CNCL_THROW(cnclDestroyComms(&comm, 1));
    return DICL_SUCCESS;
  }

  // DIPU_API diclResult_t diclCommFinalize(diclComm_t comm);

  // DIPU_API diclResult_t diclCommAbort(diclComm_t comm);

  DIPU_API diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                              const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
    convertTypeSize(count, datatype);
    CNCL_THROW(cnclAllReduce(sendbuff, recvbuff, count, cncl_data_type[datatype], cncl_op[reduceOp],
                              comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclBroadcast(const void *sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                              int root, diclComm_t comm, deviceStream_t stream) {
    convertTypeSize(count, datatype);
    CNCL_THROW(cnclBroadcast(sendbuff, recvbuff, count, cncl_data_type[datatype], root,
                              comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclAllGather(const void *sendBuf, void *recvBuf, size_t count, at::ScalarType datatype,
                              diclComm_t comm, deviceStream_t stream) {
    convertTypeSize(count, datatype);
    CNCL_THROW(cnclAllGather(sendBuf, recvBuf, count, cncl_data_type[datatype], comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream) {
    convertTypeSize(count, datatype);
    CNCL_THROW(cnclReduce(sendbuff, recvbuff, count, cncl_data_type[datatype], cncl_op[reduceOp],
                          root, comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf, uint64_t count, at::ScalarType datatype, 
                            const ReduceOp& op, diclComm_t comm, deviceStream_t stream) {
    convertTypeSize(count, datatype);
    CNCL_THROW(cnclReduceScatter(sendBuf, recvBuf, count, cncl_data_type[datatype], cncl_op[op], comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclSend(void* sendbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream){
    convertTypeSize(count, datatype);
    CNCL_THROW(cnclSend(sendbuff, count, cncl_data_type[datatype], peer, comm, stream));
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream) {
    convertTypeSize(count, datatype);
    CNCL_THROW(cnclRecv(recvbuff, count, cncl_data_type[datatype], peer, comm, stream));
    return DICL_SUCCESS;
  }

} // end namespace devapis
} // end namespace dipu
