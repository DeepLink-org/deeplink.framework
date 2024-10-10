#ifndef __PCCL_API_H__
#define __PCCL_API_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "tang_rt/driver_types.h"

#define PCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[PCCL_UNIQUE_ID_BYTES];
} pcclUniqueId;

/* Opaque handle to communicator */
typedef struct pcclComm* pcclComm_t;

/* Error enum */
typedef enum {
  pcclSuccess = 0,
  pcclUnhandledTangError = 1,
  pcclSystemError = 2,
  pcclInternalError = 3,
  pcclInvalidArgument = 4,
  pcclInvalidUsage = 5,
  pcclRemoteError = 6,
  pcclInProgress = 7,
  pcclInvalidDeviceIndex = 8,
  pccl_NUM_RESULTS
} pcclResult_t;

/* description  : Generates a unique Id with each call
 * input        : pcclUniqueId type pointer
 * output       : 0:pcclSuccess, other failure
 * note         : N/A
 */
pcclResult_t pcclGetUniqueId(pcclUniqueId* uniqueId);

/* description  : Creates a new communicator
 * input        : comm, created communicator on tang device
 *              : ndev, number of logical devices
 *              : commId, unique Id for communicator
 *              : rank, must be between 0 and ndev-1
 * output       : 0:pcclSuccess, other failure
 * note         : the func implicitly syncronizes with other ranks, so INIT OF
 * EACH RANK MUST BE CALLED IN A SEPARATE HOST THREADS to avoid deadlock.
 */
pcclResult_t pcclCommInitRank(pcclComm_t* comm, int ndev, pcclUniqueId commId,
                              int rank);

/* description  : Creates a clique of communicators
 * input        : comms, should be pre-allocated with size at least
 * ndev*sizeof(pcclComm_t) : ndev, number of logical devices : devlist, the set
 * of dev pointer, if NULL, first device to ndev used output       :
 * 0:pcclSuccess, other failure note         : This is a convenience function to
 * create a single-process communicator clique
 */
pcclResult_t pcclCommInitAll(pcclComm_t* comms, int ndev, const int* devlist);

/* description  : Frees resources associated with communicator object
 * input        : comm, the communicator
 * output       : void
 * note         : N/A
 */
pcclResult_t pcclCommDestroy(pcclComm_t comm);

/* description  : communicator abort to ask device kernel to quit
 * input        : comm, the communicator
 * output       : void
 * note         : N/A
 */
pcclResult_t pcclCommAbort(pcclComm_t comm);

/* description  : Get communicator async error
 * input        : comm, the communicator
 * output       : asyncError, the out value error
 * note         : N/A
 */
pcclResult_t pcclCommGetAsyncError(pcclComm_t comm, pcclResult_t* asyncError);

/* description  : Returns human error message
 * input        : result, the result flag
 * output       : readable error string
 * note         : N/A
 */
const char* pcclGetErrorString(pcclResult_t result);

const char* pcclGetLastError(pcclComm_t comm);

/* description  : get the number of devices in the communicator clique
 * input        : comm, the communicator
 *              : count, return value pointer
 * output       : 0:pcclSuccess, other failure
 * note         : N/A
 */
pcclResult_t pcclCommCount(const pcclComm_t comm, int* count);

/* description  : get tang device number associated with communicator
 * input        : comm, the communicator
 *              : device, return value pointer
 * output       : 0:pcclSuccess, other failure
 * note         : N/A
 */
pcclResult_t pcclCommCuDevice(const pcclComm_t comm, int* device);

/* description  : get user-ordered "rank" assocaiated with communicator
 * input        : comm, the communicator
 *              : rank, return value pointer
 * output       : 0:pcclSuccess, other failure
 * note         : N/A
 */
pcclResult_t pcclCommUserRank(const pcclComm_t comm, int* rank);

/* description  : get pccl lib version
 * input        : version, the pointers
 * output       : 0:pcclSuccess, other failure
 * note         : N/A
 */
pcclResult_t pcclGetVersion(int* version);

/* Reduction opperation selector */
typedef enum {
  pcclSum = 0,
  pcclProd = 1,
  pcclMax = 2,
  pcclMin = 3,
  pcclAvg = 4,
  pcclOpsNum = 5,
  pcclNull = pcclOpsNum
} pcclRedOp_t;

/* Data types unspported double */
typedef enum {
  pcclChar = 0,
  pcclInt8 = pcclChar,
  pcclUint8 = 1,
  pcclInt = 2,
  pcclInt32 = pcclInt,
  pcclUint32 = 3,
  pcclInt64 = 4,
  pcclUint64 = 5,
  pcclHalf = 6,
  pcclFloat16 = pcclHalf,
  pcclFloat = 7,
  pcclFloat32 = pcclFloat,
  pcclBfloat16 = 8,
  pcclTypesNum
} pcclDataType_t;

/* description  : Reduces
 * input        : sendbuff, input data buffer
 *              : recvbuff, output data buffer
 *              : count, data size
 *              : datatype, data type
 *              : op, reduce op
 *              : root, root device
 *              : comm, communicator
 *              : stream, if null,used default
 * output       : 0:pcclSuccess, other failure
 * note         : recvbuf may be NULL on all calls except for root device,
 *                sendbuff and recvbuff are assumed to reside on root device
 */
pcclResult_t pcclReduce(const void* sendbuff, void* recvbuf, size_t count,
                        pcclDataType_t datatype, pcclRedOp_t op, int root,
                        pcclComm_t comm, tangStream_t stream);

/* description  : AllReduces
 * input        : sendbuff, input data buffer
 *              : recvbuff, output data buffer
 *              : count, data size
 *              : datatype, data type
 *              : op, reduce op
 *              : comm, communicator
 *              : stream, if null,used default
 * output       : 0:pcclSuccess, other failure
 * note         : N/A
 */
pcclResult_t pcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           pcclDataType_t datatype, pcclRedOp_t op,
                           pcclComm_t comm, tangStream_t stream);

/* description  : ReducesScatter
 * input        : sendbuff, input data buffer
 *              : recvbuff, output data buffer
 *              : recvcount, i-th block data size
 *              : datatype, data type
 *              : op, reduce op
 *              : comm, communicator
 *              : stream, if null,used default
 * output       : 0:pcclSuccess, other failure
 * note         : N/A
 */
pcclResult_t pcclReduceScatter(const void* sendbuff, void* recvbuff,
                               size_t recvcount, pcclDataType_t datatype,
                               pcclRedOp_t op, pcclComm_t comm,
                               tangStream_t stream);

/* description  : Broadcast
 * input        : buff, input data buffer
 *              : count, data size
 *              : datatype, data type
 *              : root, root device
 *              : comm, communicator
 *              : stream, if null,used default
 * output       : 0:pcclSuccess, other failure
 * note         : Must be called separately for each communicator in
 * communicator clique
 */
pcclResult_t pcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           pcclDataType_t datatype, int root, pcclComm_t comm,
                           tangStream_t stream);

/* description  : AllGather
 * input        : sendbuff, input data buffer
 *              : recvbuff, output data buffer
 *              : count, data size
 *              : datatype, data type
 *              : comm, communicator
 *              : stream, if null,used default
 * output       : 0:pcclSuccess, other failure
 * note         : Must be called separately for each communicator in
 * communicator clique
 */
pcclResult_t pcclAllGather(const void* sendbuff, void* recvbuff, size_t count,
                           pcclDataType_t datatype, pcclComm_t comm,
                           tangStream_t stream);

/* description  : P2P send
 * input        : sendbuff, input data buffer
 *              : count, data size
 *              : datatype, data type
 *              : peer, the send rank index
 *              : comm, communicator
 *              : stream, if null,used default
 * output       : 0:pcclSuccess, other failure
 * note         : Must be called pcclRecv in group protect
 */
pcclResult_t pcclSend(const void* sendbuff, size_t count,
                      pcclDataType_t datatype, int peer, pcclComm_t comm,
                      tangStream_t stream);

/* description  : P2P recv
 *              : recvbuff, output data buffer
 *              : count, data size
 *              : datatype, data type
 *              : peer, recv data from rank index
 *              : comm, communicator
 *              : stream, if null,used default
 * output       : 0:pcclSuccess, other failure
 * note         : Must be called pcclSend in group protect
 */
pcclResult_t pcclRecv(void* recvbuff, size_t count, pcclDataType_t datatype,
                      int peer, pcclComm_t comm, tangStream_t stream);

pcclResult_t pcclGroupStart(void);
pcclResult_t pcclGroupEnd(void);

#ifdef __cplusplus
}
#endif
#endif  // end __PCCL_API_H__
