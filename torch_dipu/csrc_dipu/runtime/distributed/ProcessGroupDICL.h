#pragma once

#include <vector> 
#include <unordered_map>
#include <chrono>

#include <c10/core/Device.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <csrc_dipu/common.h>
#include <csrc_dipu/runtime/core/DIPUStream.h>
#include <csrc_dipu/runtime/core/DIPUEvent.h>
#include <csrc_dipu/vendor/vendorapi.h>
#include "./DICLUtils.hpp"

namespace dipu {

using Backend = c10d::Backend;
using Store = c10d::Store;
using Work = c10d::Work;
using OpType = c10d::OpType;
using BroadcastOptions = c10d::BroadcastOptions;
using AllreduceOptions = c10d::AllreduceOptions;
using ReduceOptions = c10d::ReduceOptions;
using AllgatherOptions = c10d::AllgatherOptions;
using BarrierOptions = c10d::BarrierOptions;


// Environment variable which controls whether or not wait() is blocking or
// non-blocking.
constexpr const char* DICL_BLOCKING_WAIT = "DICL_BLOCKING_WAIT";
constexpr int64_t diclSyncBusyWaitMillis = 10;

// ProcessGroupDICL implements DICLbindings for c10d.
//
// All functions of the class are expected to be called in the same order
// across all processes in the process group.  This is the only way that we
// can guarantee to match up the same calls among all processes.
//
// All DICLfunctions provided by this class are asynchronous functions. More
// specifically, each DICLcall is scheduled on a separate DIPU stream that is
// different from the current DIPU stream. This is for the purpose of
// achieving potentially concurrency and better performance. As a result,
// it is the callers' responsibilty to make sure that the DIPU stream their
// code works on needs to wait for the DICLoperation from
// this class.
//
// This can be done by calling:
//
// either WorkDICL::wait() or WorkDICL::synchronize(), both achieves the same
// functionality and are synonyms.
//
// Note that WorkDICL::isSuccess() and WorkDICL::isCompleted() will always
// return true since ProcessGroupDICL is single threaded. Every single DICL
// or DIPU failure will simply raise std::runtime_error.
//
// Therefore, WorkDICL::exception() is not supported since isSuccess() always
// returns true.
//
// Also note that WorkDICL::finishedDIPUExecution() is a helper function only
// provided by ProcessGroupDICL to check if the DICL operation of WorkDICL has
// finished execution on the DIPU (not just scheduled).
//
// Example on using the DICL process group
//
//   ProcessGroupDICL pg(store, rank, size);
//   std::shared_ptr<WorkDICL> work = pg.allreduce(tensors);
//
//   // At this point, DICL kernel has already by queued successfully
//   // Now, let current stream wait for the DICL to finish, originally this function is
//   // async operation as well, but currently DIPU is sync.
//
//   work->wait()
//
//   // Now continue on other work in the current stream.

// not support gather/scatter/reduce_scatter/ all _coalesced func & _base func now, 
// If needed in the future, we will add
class DIPU_API ProcessGroupDICL : public Backend {
 public:
  class WorkDICL : public Work {
  public:
    // Constructor takes a list of DIPU devices
    WorkDICL(const std::vector<at::Device>& devices); // NOLINT

    virtual ~WorkDICL();

    // Checks if request has completed. In this specific case of DICL, it checks
    // if the DICL operation has completed on the DIPU in its own DICL queue.
    // Non-blocking operation.
    bool isCompleted() override;

    bool isSuccess() const override;

    // Same as calling synchronize() for DICL work.
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    // Let current stream wait on the completing of the DICL work
    // Throws on exceptions
    void synchronize() override;

  protected:
    // The cached list of DIPU devices to operate on
    std::vector<at::Device> devices_;

    // The DIPU events tracking this work item on DIPU device
    std::vector<dipu::DIPUEvent> events_;

    // The DICL communicators used for this work item.
    std::vector<std::shared_ptr<DICLComm>> diclComms_;


    // Clone of blockingWait_ from ProcessGroupDICL.
    bool blockingWait_ = false;

    // Tensors used for barrier op
    std::vector<at::Tensor> barrierTensors_;

    // Just checks whether DIPU execution has completed, without modifying
    // exception_ptr.
    bool finishedDICLExecutionInternal() const;

    // Clone of opTimeout_ from ProcessGroupHCCL.
    std::chrono::milliseconds opTimeout_;

    // Time point representing when the work started.
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

  private:
      friend class ProcessGroupDICL;
  };

  struct DIPU_API Options : Backend::Options {
    // NOTE: timeout in ProcessGroupNCCL::Options denote the timeout for
    // operations. This is only used when blockingWait_ is enabled.
    explicit Options() : Backend::Options(DICL_BACKEND_NAME) {}
  };

  // If you wish to create multiple process groups, each with a potentially
  // different rank and size, you can do so by passing a new store instance
  // to each one. If you have only a single store object, you can
  // use the `c10d::PrefixStore` to derive scoped instances.
  // This is also what the Python API in torch.distributed does.
  //
  // The process group instance keeps a reference to the store because
  // it may be used long after the constructor runs. In fact, the constructor
  // doesn't create any DICL communicators. A single DICL communicator can
  // only be used on a specific set of devices, and are therefore created
  // on-demand when a collective runs. If another collective is executed later,
  // against a different set of devices, the process group creates another DICL
  // communicator. These DICL communicators are cached and reused if possible.
  ProcessGroupDICL(const c10::intrusive_ptr<Store>& store, int rank, int size);

  virtual ~ProcessGroupDICL();

  c10::intrusive_ptr<Work> broadcast(std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> reduce(std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors, const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors,
      int dstRank, int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors,
      int srcRank, int tag) override;

  c10::intrusive_ptr<Work> barrier(const BarrierOptions& opts = BarrierOptions()) override;

 protected:
  // different device may need extend this func to do device specific check
  virtual void checkDeviceTensors(const std::vector<at::Tensor>& tensors);
  virtual c10::intrusive_ptr<ProcessGroupDICL::WorkDICL> initWork(std::vector<at::Device> devices);

  // Helper that broadcasts DICL clique ID to all ranks through the store
  virtual void broadcastUniqueID(commUniqueId_t uniqueId, bool isSingleP2POp,
                              const std::string& p2pKey, int p2pRank);

  // Helper that either looks up the cached DICL communicators or creates
  // a new set of DICL communicators as a cache entry
  virtual std::vector<std::shared_ptr<DICLComm>>& getDICLComms(
        const std::string& devicesKey, const std::vector<at::Device>& devices,
        OpType opType, int p2pRank = 0, bool isSendRecvSelf = false);

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input, std::vector<at::Tensor>& output, Fn fn, OpType opType);

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& input, std::vector<at::Tensor>& output, Fn fn,
      PreProcess pre, PostProcess post, OpType opType);

  // The store is used to broadcast the DICL unique ID of rank 0.
  c10::intrusive_ptr<Store> store_;

  // The number of DICL communicators that have been created during
  // the lifetime of this process group. This sequence number is
  // used to scope keys used in the store.
  uint64_t diclCommCounter_{0};

  // The DICL communicator that the process group has cached.
  // The key is a list of DIPU devices that an operation is operating on
  // The DIPU devices are stored in a device sequence and the cache DICL
  // communicator is associated with this DIPU device sequence
  //
  // e.g. If the process group op only uses device 0, then the value of
  // the used device string stored (value of the hashmap) would be "0".
  //    
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 1, 2, 3, 4, 5, 6, 7 separately,
  //      then the value of the used device string (key) stored would be
  //      "0,1,2,3,4,5,6,7"
  //
  //      If the process group op uses device 0 - 7 and the each tensor of the
  //      input tensor list is on device, 0, 4, 5, 6, 7, 1, 2, 3 separately,
  //      then the value of the used device string stored would be
  //      "0,4,5,6,7,1,2,3"
  //
  //      Note that the order of the device for the tensor list matters.
  std::unordered_map<std::string, std::vector<std::shared_ptr<DICLComm>>> devDICLCommsMap_;

  // Mutex to guard devDICLCommMap_.
  std::mutex devDICLCommMapLock_;

  // The DIPU queues used by DICL kernels
  std::unordered_map<std::string, std::vector<DIPUStream>> diclStreams_;

  // The DIPU events used to sync DICL queues
  std::unordered_map<std::string, std::vector<DIPUEvent>> diclEvents_;

  // Device Indexes used for all collectives in this group
  std::set<int> usedDeviceIdxs_;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
  bool blockingWait_ = false;

  std::chrono::milliseconds opTimeout_;

  // Time point representing when the work started.
  std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

};

c10::intrusive_ptr<c10d::Backend> createProcessGroupDICL(
      const c10::intrusive_ptr<::c10d::Store> &store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout) ;

}  // namespace dipu