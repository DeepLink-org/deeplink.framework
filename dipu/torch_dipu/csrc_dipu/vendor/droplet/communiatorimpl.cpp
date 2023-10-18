#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {


  const int DICL_UNIQUE_ID_BYTES_SIZE = 0;

  DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclGetUniqueId(pucclUniqueId* uniqueId) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, pucclUniqueId uniqueId,
                                          int rank, int localDeviceId) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                              const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclBroadcast(const void *sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                              int root, diclComm_t comm, deviceStream_t stream) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclAllGather(const void *sendBuf, void *recvBuf, size_t count, at::ScalarType datatype,
                              diclComm_t comm, deviceStream_t stream) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream) {
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf, size_t recvCount, at::ScalarType dataType, 
                                  const ReduceOp& op, diclComm_t comm, deviceStream_t stream) {
    throw std::runtime_error("mlu Not implement diclReduceScatter");
  }

  DIPU_API diclResult_t diclSend(void* sendbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream){
    return DICL_SUCCESS;
  }

  DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count, at::ScalarType datatype, int peer,
                          diclComm_t comm, deviceStream_t stream) {
    return DICL_SUCCESS;
  }

} // end namespace devapis
} // end namespace dipu
