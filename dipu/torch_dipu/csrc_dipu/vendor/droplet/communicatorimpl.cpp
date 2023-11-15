#include <stdexcept>
#include <string>

#include <c10/core/ScalarType.h>
#ifdef USE_PCCL
#include <pccl.h>
#endif // USE_PCCL
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

namespace devapis {

#ifdef USE_PCCL

#define LINE_HELPER1(x) #x
#define LINE_HELPER2(x) LINE_HELPER1(x)
#define LOCATION __FILE__ " : " LINE_HELPER2(__LINE__)

const int DICL_UNIQUE_ID_BYTES_SIZE = PCCL_UNIQUE_ID_BYTES;

// PCCL reduce-op mapping
static const std::map<ReduceOp::RedOpType, pcclRedOp_t> toPcclReduceOp = {
    {ReduceOp::MIN, pcclMin},
    {ReduceOp::MAX, pcclMax},
    {ReduceOp::SUM, pcclSum},
    {ReduceOp::PRODUCT, pcclProd},
    {ReduceOp::AVG, pcclAvg},
};

// TODO: find a better function to get reduce-op's name
#define RedOpTypeToPcclRedOp_t(op_type, pccl_op)                                                      \
pcclRedOp_t pccl_op;                                                                                  \
{                                                                                                     \
    auto p = toPcclReduceOp.find(op_type);                                                            \
    if (p == toPcclReduceOp.end()) {                                                                  \
        std::string err = "Unsupported reduce op " + std::to_string(op_type) + " at: " LOCATION "\n"; \
        throw std::runtime_error(err);                                                                \
    }                                                                                                 \
    pccl_op = p->second;                                                                              \
}

// PCCL dtype mapping
static const std::map<at::ScalarType, pcclDataType_t> toPcclDataType = {
    {at::kChar, pcclInt8},
    {at::kByte, pcclUint8},
    {at::kFloat, pcclFloat},
    // TODO: PCCL not support double now
    // {at::kDouble, pcclDouble},
    {at::kInt, pcclInt32},
    {at::kLong, pcclInt64},
    {at::kHalf, pcclHalf},
    {at::kBool, pcclUint8},
    {at::kBFloat16, pcclBfloat16},
};

#define ScalarTypeToPcclDataType_t(scalar_type, pccl_data_type)                                     \
pcclDataType_t pccl_data_type;                                                                      \
{                                                                                                   \
    auto p = toPcclDataType.find(scalar_type);                                                      \
    if (p == toPcclDataType.end()) {                                                                \
        std::string err = std::string("Not supported ScalarType ") + c10::toString(scalar_type) +   \
            " at: " LOCATION "\n";                                                                  \
        throw std::runtime_error(err);                                                              \
    }                                                                                               \
    pccl_data_type = p->second;                                                                     \
}

// Macro to print and abort on a non-successful PCCL return value.
#define CALL_PCCL(expr)                                                                             \
do {                                                                                                \
    pcclResult_t result = expr;                                                                     \
    if (result != pcclSuccess) {                                                                    \
        std::string err = "PCCL error at: " LOCATION ", return code=" + std::to_string(result) +    \
            ", err_str:"  + pcclGetErrorString(result);                                             \
        throw std::runtime_error(err);                                                              \
    }                                                                                               \
} while (0)

DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
    pcclResult_t pcclAsyncErr;
    CALL_PCCL(pcclCommGetAsyncError(comm, &pcclAsyncErr));
    // shuold we return pcclInProgress as success or not?
    if (pcclAsyncErr != pcclSuccess) {
        return DICL_ERR_UNDEF;
    } else {
        return DICL_SUCCESS;
    }
}

DIPU_API diclResult_t diclGetUniqueId(commUniqueId* uniqueId) {
    CALL_PCCL(pcclGetUniqueId(uniqueId));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, commUniqueId uniqueId,
                                       int rank, int localDeviceId) {
    CALL_PCCL(pcclCommInitRank(comm, nranks, uniqueId, rank));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
    CALL_PCCL(pcclCommDestroy(comm));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
    ScalarTypeToPcclDataType_t(datatype, pcclDataType);
    RedOpTypeToPcclRedOp_t(reduceOp, pcclReduceOp);
    CALL_PCCL(pcclAllReduce(sendbuff, recvbuff, count, pcclDataType, pcclReduceOp, comm, stream));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclBroadcast(const void *sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            int root, diclComm_t comm, deviceStream_t stream) {
    ScalarTypeToPcclDataType_t(datatype, pcclDataType);
    CALL_PCCL(pcclBroadcast(sendbuff, recvbuff, count, pcclDataType, root, comm, stream));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclAllGather(const void *sendBuf, void *recvBuf, size_t count, at::ScalarType datatype,
                            diclComm_t comm, deviceStream_t stream) {
    ScalarTypeToPcclDataType_t(datatype, pcclDataType);
    CALL_PCCL(pcclAllGather(sendBuf, recvBuf, count, pcclDataType, comm, stream));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                          const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream) {
    ScalarTypeToPcclDataType_t(datatype, pcclDataType);
    RedOpTypeToPcclRedOp_t(reduceOp, pcclReduceOp);
    CALL_PCCL(pcclReduce(sendbuff, recvbuff, count, pcclDataType, pcclReduceOp, root, comm, stream));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf, size_t recvCount, at::ScalarType datatype,
                                const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
    ScalarTypeToPcclDataType_t(datatype, pcclDataType);
    RedOpTypeToPcclRedOp_t(reduceOp, pcclReduceOp);
    CALL_PCCL(pcclReduceScatter(sendBuf, recvBuf, recvCount, pcclDataType, pcclReduceOp, comm, stream));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclSend(void* sendbuff, size_t count, at::ScalarType datatype, int peer,
                        diclComm_t comm, deviceStream_t stream){
    ScalarTypeToPcclDataType_t(datatype, pcclDataType);
    CALL_PCCL(pcclSend(sendbuff, count, pcclDataType, peer, comm, stream));
    return DICL_SUCCESS;
}

DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count, at::ScalarType datatype, int peer,
                        diclComm_t comm, deviceStream_t stream) {
    ScalarTypeToPcclDataType_t(datatype, pcclDataType);
    CALL_PCCL(pcclRecv(recvbuff, count, pcclDataType, peer, comm, stream));
    return DICL_SUCCESS;
}

#else // USE_PCCL

const int DICL_UNIQUE_ID_BYTES_SIZE = 0;

DIPU_API diclResult_t diclGetCommAsyncError(diclComm_t comm) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclGetUniqueId(pcclUniqueId* uniqueId) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclCommInitRank(diclComm_t* comm, int nranks, pcclUniqueId uniqueId,
                                        int rank, int localDeviceId) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclCommDestroy(diclComm_t comm) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclAllReduce(const void *sendbuff, void *recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclBroadcast(const void *sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            int root, diclComm_t comm, deviceStream_t stream) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclAllGather(const void *sendBuf, void *recvBuf, size_t count, at::ScalarType datatype,
                            diclComm_t comm, deviceStream_t stream) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclReduce(const void* sendbuff, void* recvbuff, size_t count, at::ScalarType datatype,
                            const ReduceOp& reduceOp, int root, diclComm_t comm, deviceStream_t stream) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclReduceScatter(void *sendBuf, void *recvBuf, size_t recvCount, at::ScalarType datatype,
                                const ReduceOp& reduceOp, diclComm_t comm, deviceStream_t stream) {
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclSend(void* sendbuff, size_t count, at::ScalarType datatype, int peer,
                        diclComm_t comm, deviceStream_t stream){
    return DICL_ERR_UNDEF;
}

DIPU_API diclResult_t diclRecv(void* recvbuff, size_t count, at::ScalarType datatype, int peer,
                        diclComm_t comm, deviceStream_t stream) {
    return DICL_ERR_UNDEF;
}

#endif // USE_PCCL

} // end namespace devapis

} // end namespace dipu
