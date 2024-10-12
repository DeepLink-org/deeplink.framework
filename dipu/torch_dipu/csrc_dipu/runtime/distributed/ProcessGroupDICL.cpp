// Copyright (c) 2023, DeepLink.
#include "ProcessGroupDICL.h"

#include <fstream>
#include <mutex>
#include <utility>
#include <vector>

#include <ATen/core/TensorBody.h>
#include <ATen/ops/cat.h>
#include <ATen/record_function.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <c10/util/typeid.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/torch.h>

#include "csrc_dipu/aten/ops/NodispatchUtils.hpp"
#include "csrc_dipu/profiler/profiler.h"
#include "csrc_dipu/runtime/core/DIPUGuard.h"
#include "csrc_dipu/runtime/core/DIPUStream.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/runtime/devproxy/diclproxy.h"
#include "csrc_dipu/utils/helpfunc.hpp"
#include <csrc_dipu/vendor/vendorapi.h>

namespace dipu {

using std::pair;
using RaiseInvalidArgFunc = std::function<void(const std::string&)>;

namespace {

// Get the list of devices from list of tensors, collective comm always use all
// ranks, so no rank prefix required in key.
std::string getDeviceIds(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

pair<int, int> mapPGRank2P2P(int myRank, int peer) {
  // ProcessGroupNCCL support send/recv self, but that seems only work with
  // ncclGroup?
  TORCH_CHECK(myRank != peer,
              "Invalid destination rank: should not be "
              "the same as rank of the current process.");
  pair<int, int> p2pRanks;
  // self p2p rank
  p2pRanks.first = myRank <= peer ? 0 : 1;
  // p2p target rank
  p2pRanks.second = 1 - p2pRanks.first;
  return p2pRanks;
}

// Get p2p sorted ranks as key, p2p only support 1 device tensor at a time and
// one comm endpoint can bind with either device. so use rank as comm key is
// enough.
std::string getP2PRankIds(int myRank, int peer,
                          const std::vector<at::Device>& devices) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  return std::to_string(lowRank) + ":" + std::to_string(highRank);
}

std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

void syncStreams(std::vector<std::shared_ptr<DICLComm>>& comms) {
  for (auto& comm : comms) {
    comm->preSyncStream();
  }
}

RaiseInvalidArgFunc getInvalidArgumentFunc(const std::string& prefix) {
  return [&](const std::string& msg) { TORCH_CHECK(false, prefix + msg) };
}

// Check function for root rank in scatter and gather op:
// We assume that tensors has only one element and tensors[0] has numRanks
// elements. Dtype and shape of elements in tensors[0] should be the same as
// other.
void checkGatherScatterRootRank(
    const std::vector<std::vector<at::Tensor>>& tensors,
    const at::Tensor& other, int numRanks,
    const RaiseInvalidArgFunc& raise_invalid_arg_func) {
  if (tensors.size() != 1) {
    std::stringstream ss;
    ss << "requires a single-element list containing a list with " << numRanks
       << " tensors.";
    raise_invalid_arg_func(ss.str());
  }
  if (tensors[0].size() != static_cast<size_t>(numRanks)) {
    std::stringstream ss;
    ss << "incorrect list size " << tensors[0].size()
       << ". The list size should be " << numRanks
       << ", same as size of the process group.";
    raise_invalid_arg_func(ss.str());
  }

  const auto& options = other.options();
  const auto& sizes = other.sizes();
  c10d::assertTypeAndSizesMatch(raise_invalid_arg_func, tensors[0], options,
                                sizes);
}

}  // anonymous namespace

// start WorkStore

class WorkStore {
  struct WorkInfo {
    DIPUEvent startEvent_;
    DIPUEvent endEvent_;
    int rank_;
    int comm_size_;
  };

 public:
  void setUid(const std::vector<uint8_t>& uidVec) { uniqueidVec_ = uidVec; }

  size_t recordStart(const DIPUStream& stream, int rank, int comm_size) {
    std::lock_guard<std::mutex> lock(mtx_);
    info_vec_.push_back(WorkInfo());
    size_t index = info_vec_.size() - 1;
    info_vec_[index].startEvent_.record(stream, false);
    info_vec_[index].rank_ = rank;
    info_vec_[index].comm_size_ = comm_size;

    return index;
  }

  void recordEnd(const DIPUStream& stream, size_t index) {
    std::lock_guard<std::mutex> lock(mtx_);
    info_vec_[index].endEvent_.record(stream, false);
  }

  void dump(std::string& path) {
    for (auto& wi : info_vec_) {
      wi.endEvent_.synchronize();
      float duration = wi.startEvent_.elapsed_time(wi.endEvent_);
      std::ostringstream oss;
      oss << "PG uniqueId = ";
      for (int i = 0; i < 32; ++i) {
        oss << static_cast<int>(uniqueidVec_[i]);
      }
      oss << ", comm_size = " << wi.comm_size_ << ", duration = " << duration
          << std::endl;
      std::string filePath = path + "/rank_" + std::to_string(wi.rank_);
      std::ofstream outFile(filePath, std::ios::app);
      outFile << oss.str();
    }

    info_vec_.clear();
  }

 private:
  std::vector<WorkInfo> info_vec_;
  std::mutex mtx_;
  std::vector<uint8_t> uniqueidVec_;
};

// end WorkStore

std::vector<std::shared_ptr<WorkStore>> global_stores;

void dumpInfo(std::string& path) {
  for (auto p : global_stores) {
    p->dump(path);
  }
}

// start WorkDICL

// currently DICL do not support error check
bool ProcessGroupDICL::WorkDICL::isCompleted() {
  return finishedDICLExecutionInternal();
}

// currently DICL do not support error check
bool ProcessGroupDICL::WorkDICL::isSuccess() const {
  return finishedDICLExecutionInternal();
}

bool ProcessGroupDICL::WorkDICL::finishedDICLExecutionInternal() const {
  return std::all_of(workEvents_.begin(), workEvents_.end(),
                     [](const DIPUEvent& e) { return e.query(); });
}

// record post work event on communicator stream
void ProcessGroupDICL::WorkDICL::record() {
  for (auto i = 0; i < workEvents_.size(); i++) {
    workEvents_[i].record(diclComms_[i]->diclStream_);
  }
}

void ProcessGroupDICL::WorkDICL::synchronize() {
  for (auto i = 0; i < workEvents_.size(); i++) {
    auto currentStream =
        dipu::getCurrentDIPUStream(diclComms_[i]->device_.index());
    // Block the current stream(calculate stream) on the DICL comm stream event
    workEvents_[i].wait(currentStream);
  }

  // In case of blocking, wait for the operation to complete.
  if (blockingWait_) {
    // Wait for the operation to complete.
    while (!isCompleted()) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(
              currentTimepoint - workStartTime_) > opTimeout_) {
        throw std::runtime_error("Operation timed out!");
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(diclSyncBusyWaitMillis));
    }
  }

  // Device synchronize only after we've completed timeout checks.
  // only barrier() call this
  if (barrier_) {
    // If we use the work to do barrier, we should block here
    for (auto& comm : diclComms_) {
      DIPUGuard dipuGuard(comm->device_);
      devproxy::syncDevice();
    }
  }
}

// Same as calling synchronize().
bool ProcessGroupDICL::WorkDICL::wait(std::chrono::milliseconds timeout) {
  synchronize();
  return true;
}

std::vector<at::Tensor> ProcessGroupDICL::WorkDICL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupDICL::WorkDICL::getFuture() {
  return future_;
}

// end WorkDICL

ProcessGroupDICL::ProcessGroupDICL(const c10::intrusive_ptr<Store>& store,
                                   int rank, int size)
    : c10d::Backend(rank, size),
      store_(store),
      pWstore_(std::make_shared<WorkStore>()) {
  global_stores.push_back(pWstore_);
  char* blockingWait = getenv(DICL_BLOCKING_WAIT);
  try {
    if (blockingWait != nullptr) {
      auto val = std::stoi(blockingWait);
      if (val == 1) {
        // Make wait() and synchronize() a blocking call.
        blockingWait_ = true;
      } else if (val != 0) {
        throw std::runtime_error("Invalid value for environment variable: " +
                                 std::string(DICL_BLOCKING_WAIT));
      }
    }
  } catch (std::exception& e) {
    throw std::runtime_error("Invalid value for environment variable: " +
                             std::string(DICL_BLOCKING_WAIT));
  }
}

ProcessGroupDICL::~ProcessGroupDICL() = default;

void ProcessGroupDICL::broadcastUniqueID(commUniqueId* uniqueId,
                                         const std::string& storeKey,
                                         int commRank) {
  // For collective operations:
  // For every DICL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple DICL communicators, so we use a sequence
  // number to differentiate between them.
  // For point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  if (commRank == 0) {
    auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(uniqueId),
                                    reinterpret_cast<uint8_t*>(uniqueId) +
                                        devapis::DICL_UNIQUE_ID_BYTES_SIZE);
    pWstore_->setUid(vec);
    store_->set(storeKey, vec);
  } else {
    auto vec = store_->get(storeKey);
    if (vec.size() != devapis::DICL_UNIQUE_ID_BYTES_SIZE) {
      throw std::runtime_error(
          "Unexpected DICL unique ID length received "
          "from the store");
    }
    pWstore_->setUid(vec);
    std::memcpy(uniqueId, vec.data(), vec.size());
  }
}

std::vector<std::shared_ptr<DICLComm>>& ProcessGroupDICL::getDICLComms(
    const std::string& localCommsKey, const std::vector<at::Device>& devices,
    int commsRank, OpType opType) {
  // Sanity check
  if (localCommsKey.empty()) {
    throw std::runtime_error(
        "Not able to create/get the DICL Communicator since "
        "the DIPU devices are not known");
  }
  {
    std::lock_guard<std::mutex> lock(devDICLCommMapLock_);
    if (devDICLCommsMap_.find(localCommsKey) != devDICLCommsMap_.end()) {
      // Reuse the cached communicator if there is one.
      return devDICLCommsMap_[localCommsKey];
    }
  }
  // not cached, create a new entry
  std::vector<std::shared_ptr<DICLComm>> diclComms;
  int devSize = static_cast<int>(devices.size());
  diclComms.resize(devSize);
  int deviceWorldSize = isP2POp(opType, false) ? 2 : getSize() * devSize;

  commUniqueId diclID;
  if (commsRank == 0) {
    devproxy::diclGetUniqueId(&diclID);
  }
  std::string bcastKey = isP2POp(opType, false)
                             ? localCommsKey
                             : std::to_string(diclCommCounter_++);
  broadcastUniqueID(&diclID, bcastKey, commsRank);

  OptionalDIPUGuard dipuGuard;

  for (int i = 0; i < devSize; i++) {
    int deviceCommRank =
        isP2POp(opType, false) ? commsRank : getRank() * devSize + i;
    dipuGuard.reset_device(devices[i]);
    // use pool stream, not current stream
    auto commStream = getDIPUStreamFromPool(devices[i].index());
    diclComms[i] =
        DICLComm::create(deviceWorldSize, deviceCommRank, diclID, commStream);
  }

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(devDICLCommMapLock_);
  // Move the DICL resource to cache
  devDICLCommsMap_.emplace(localCommsKey, std::move(diclComms));
  return devDICLCommsMap_[localCommsKey];
}

namespace {

// Ref:
// https://github.com/pytorch/pytorch/blob/f2d7f235a684c593f5a1ff2ca0b47b47274bfe85/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L2242-L2261
void check_device_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false  // whether operation is a P2P operation
) {
  if (!dipu::isDeviceTensor(tensor) || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be DIPU and dense");
  }
  // Skip the following requirements for P2P operations
  if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
    if (p2p) {
      TORCH_WARN_ONCE(
          "Detected non-contiguous tensor in P2P operations. It is user "
          "responsibility to guarantee that source and destination tensors "
          "have the same contiguity format.");
    } else {
      C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
    }
  }
}

// Check that all `tensors'
void checkDeviceTensors(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    TORCH_CHECK(false, "Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(devproxy::getDeviceCount())) {
    TORCH_CHECK(
        false,
        "Tensor list mustn't be larger than the number of available DIPUs");
  }
  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& tensor : tensors) {
    if (!dipu::isDeviceTensor(tensor) ||
        !tensor.is_non_overlapping_and_dense()) {
      TORCH_CHECK(false, "Tensors must be DIPU and non-overlapping and dense");
    }
    if (tensor.scalar_type() != first.scalar_type()) {
      TORCH_CHECK(false, "Tensors must have identical type");
    }
    if (tensor.sizes() != first.sizes()) {
      TORCH_CHECK(false, "Tensors must have identical size");
    }
    if (tensor.strides() != first.strides()) {
      TORCH_CHECK(false, "Tensors must have identical strides");
    }
    const auto inserted = usedDevices.insert(tensor.get_device()).second;
    if (!inserted) {
      TORCH_CHECK(false, "Tensors must be on distinct DIPU devices");
    }
  }
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other, size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error(
        "Tensor list operands to scatter/gather must have the same length");
  }
  const auto num_devices = tensor_lists.size();
  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (auto i = size_t{}; i < num_devices; ++i) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error(
          "Tensor list input to scatter/gather must match number of collective"
          " participants");
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error(
          "Corresponding input/output tensors to scatter/gather must all reside"
          " on the same device");
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error(
            "All tensor operands to scatter/gather must have the same size");
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = c10d::newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

template <bool RecordDest, typename Dest, typename Src>
void copyInCommStream(std::shared_ptr<DICLComm>& diclComm, const Dest& dest,
                      const Src& src, int nums) {
  auto diclStream = diclComm->diclStream_;
  DIPUStreamGuard guard(diclStream.unwrap());
  for (size_t j = 0; j < nums; ++j) {
    dest[j].copy_(src[j], true);
    if (RecordDest) {
      dipu::recordStream(dest[j], diclStream);
    } else {
      dipu::recordStream(src[j], diclStream);
    }
  }
}

void copyInCurrentStream(std::shared_ptr<DICLComm>& diclComm,
                         const std::vector<at::Tensor>& dest,
                         const at::Tensor& src) {
  auto diclStream = diclComm->diclStream_;
  auto currStream = dipu::getCurrentDIPUStream(diclStream.device_index());
  diclComm->preCopyEvent_.record(diclStream);
  // copy after comm finish, loss concurrency,assume all dest finish in one comm
  // op
  diclComm->preCopyEvent_.wait(currStream);
  for (int64_t j = 0; j < dest.size(); ++j) {
    dest[j].copy_(src[j], true);
  }
}
}  // namespace

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupDICL::doComm(
    std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs,
    std::vector<std::shared_ptr<DICLComm>>& diclComms,
    const std::vector<at::Device>& devices, Fn fn, PreProcess pre,
    PostProcess post, OpType opType) {
  // First let DICL streams wait for input tensors allocation streams
  syncStreams(diclComms);
  auto work = c10::make_intrusive<ProcessGroupDICL::WorkDICL>(
      diclComms, blockingWait_, opTimeout_);

  size_t eventIndex;
  if (opType == OpType::ALLREDUCE) {
    eventIndex =
        pWstore_->recordStart(diclComms[0]->diclStream_, this->rank_,
                              inputs[0].element_size() * inputs[0].numel());
  }

  OptionalDIPUGuard dipuGuard;
  pre(diclComms);

  for (size_t i = 0; i < inputs.size(); ++i) {
    dipuGuard.reset_device(diclComms[i]->device_);

    // need add adapter to handle int64/double! camb not support double
    fn(inputs[i], outputs[i], diclComms[i]->rawComm(),
       diclComms[i]->diclStream_);

    dipu::recordStream(inputs[i], diclComms[i]->diclStream_);
    if (outputs[i].has_storage() &&
        (!inputs[i].has_storage() ||
         inputs[i].storage().data_ptr().get() !=
             outputs[i].storage().data_ptr().get())) {
      dipu::recordStream(outputs[i], diclComms[i]->diclStream_);
    }

    // mock comm with just copy, used in standalone test.
    // DIPUStreamGuard guard(diclComms[i]->diclStream_.unwrap());
    // outputs[i].copy_(inputs[i], false);
  }

  post(diclComms);

  if (opType == OpType::ALLREDUCE) {
    pWstore_->recordEnd(diclComms[0]->diclStream_, eventIndex);
  }

  work->record();

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);
  // todo:: dipu need support multistream guard & remove
  // work->workEvents_(future already has events ).
  {
    DIPUStreamGuard guard(diclComms[0]->diclStream_.unwrap());

    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }
  return work;
}

// std::function< diclResult_t(at::Tensor&, at::Tensor&, DiclComm, DIPUStream&)
// > enhance: need change template params to lamada, make collective() func
// overridable by sub class
template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupDICL::collective(
    std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs, Fn fn,
    PreProcess pre, PostProcess post, OpType opType) {
  const auto devices = getDeviceList(inputs);

  TORCH_CHECK(devices.size() == 1,
              "dipu support one device per process only, nccl multidevices use "
              "ncclGroupStart/End, ",
              "but we cannot support group based comm now.");

  const auto localCommsKey = getDeviceIds(devices);

  // collective use PG.rank_ as comsBaseRank
  auto diclComms = getDICLComms(localCommsKey, devices, this->rank_, opType);
  return doComm(inputs, outputs, diclComms, devices, fn, pre, post, opType);
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupDICL::collective(
    std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs, Fn fn,
    OpType opType) {
  return collective(
      inputs, outputs, fn, [](std::vector<std::shared_ptr<DICLComm>>&) {},
      [](std::vector<std::shared_ptr<DICLComm>>&) {}, opType);
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupDICL::pointToPoint(
    std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs,
    int peerRank, Fn fn, PreProcess pre, PostProcess post, OpType opType) {
  const auto devices = getDeviceList(inputs);
  auto p2pPair = mapPGRank2P2P(rank_, peerRank);
  // pytorch nccl has same problem but not check
  TORCH_CHECK(devices.size() == 1,
              "DICL P2P comm does not support multi-device tensor input");

  // pytorch nccl always create new comm when new send/recv pair appear, here we
  // follow this behavior. However, It's also works well by using the default
  // collective commms to do pair comm, which cost lower resource in very big
  // group size but may cause different pairs block in same stream.
  const auto localCommsKey = getP2PRankIds(rank_, peerRank, devices);

  // p2p use self p2pRank as commsRank, one commsRank corresponds to one device
  auto& diclComms = getDICLComms(localCommsKey, devices, p2pPair.first, opType);

  return doComm(inputs, outputs, diclComms, devices, fn, pre, post, opType);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  // inplace in = out, every rank use both in&out.
  checkDeviceTensors(tensors);
  std::vector<at::Tensor> tensors_cp{tensors};
  return collective(
      tensors_cp, tensors_cp,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclAllreduce", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclAllreduce", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclAllReduce(input.data_ptr(), output.data_ptr(),
                                       static_cast<size_t>(input.numel()),
                                       input.scalar_type(), opts.reduceOp, comm,
                                       stream.rawstream());
      },
      [&](std::vector<std::shared_ptr<DICLComm>>& comms) {
        if (dicl_hook::allReducePreFn) {
          dicl_hook::allReducePreFn(comms, tensors, tensors_cp);
        }
      },
      [&](std::vector<std::shared_ptr<DICLComm>>& comms) {
        if (dicl_hook::allReducePostFn) {
          dicl_hook::allReducePostFn(comms, tensors_cp, tensors);
        }
      },
      OpType::ALLREDUCE);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  checkDeviceTensors(tensors);
  // inplace in = out, only rootRank use in.
  return collective(
      tensors, tensors,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclBroadcast", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclBroadcast", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        // only one root (root rank root device)
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        return devproxy::diclBroadcast(
            input.data_ptr(), input.data_ptr(),
            static_cast<size_t>(input.numel()), input.scalar_type(),
            static_cast<int>(root), comm, stream.rawstream());
      },
      OpType::BROADCAST);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  // inplace in = out, only rootRank use out.
  checkDeviceTensors(tensors);

  auto tensor = tensors.back();
  int dev_in_group = 0;
  std::vector<at::Tensor> tensors_cp{tensors};
  return collective(
      tensors_cp, tensors_cp,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclReduce", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclReduce", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
        return devproxy::diclReduce(
            input.data_ptr(), output.data_ptr(),
            static_cast<size_t>(input.numel()), input.scalar_type(),
            opts.reduceOp, static_cast<int>(root), comm, stream.rawstream());
      },
      [&](std::vector<std::shared_ptr<DICLComm>>& comms) {
        if (dicl_hook::reducePreFn) {
          dicl_hook::reducePreFn(comms, tensors, tensors_cp);
        }
      },
      [&](std::vector<std::shared_ptr<DICLComm>>& comms) {
        if (dicl_hook::reducePostFn) {
          dicl_hook::reducePostFn(comms, tensors_cp, tensors);
        }
      },
      OpType::REDUCE);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs, const GatherOptions& opts) {
  // output = input * ranks, no inplace, input = output[rank]
  static const auto raise_invalid_arg_func =
      getInvalidArgumentFunc("ProcessGroupDICL::gather: ");
  int numRanks = getSize();
  int curRank = getRank();
  int rootRank = static_cast<int>(opts.rootRank);
  c10d::assertRootRank(raise_invalid_arg_func, rootRank, numRanks);
  checkDeviceTensors(inputs);
  c10d::assertSingleElementInput(raise_invalid_arg_func, inputs);
  auto input = inputs.back();
  std::vector<at::Tensor> outputTensors;

  if (curRank == rootRank) {
    checkGatherScatterRootRank(outputs, input, numRanks,
                               raise_invalid_arg_func);
    outputTensors = outputs[0];
  } else {
    if (!outputs.empty()) {
      raise_invalid_arg_func("requires empty output on non-root");
    }
    outputTensors = {};
    outputTensors.emplace_back();
  }

  return collective(
      inputs, outputTensors,
      [&](at::Tensor& /* unused */, at::Tensor& /* unused */, diclComm_t comm,
          DIPUStream& stream) {
        std::vector<void*> output_ptr_vec = {};
        if (curRank == rootRank) {
          output_ptr_vec.reserve(outputTensors.size());
          for (auto& outputTensor : outputTensors) {
            output_ptr_vec.push_back(outputTensor.data_ptr());
            dipu::recordStream(outputTensor, stream);
          }
        }

        RECORD_FUNCTION("DiclGather", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclGather", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        // since param `recvbuf` is only used in root rank process in
        // diclGather, we can pass output_ptr_vec.data() to it
        return devproxy::diclGather(input.data_ptr(), output_ptr_vec.data(),
                                    static_cast<size_t>(input.numel()),
                                    input.scalar_type(), rootRank, curRank,
                                    numRanks, comm, stream.rawstream());
      },
      OpType::GATHER);
}

std::string_view ProcessGroupDICL::getCommName(
    const at::DeviceIndex device_index) {
  auto device = at::Device(dipu::DIPU_DEVICE_TYPE, device_index);
  std::vector<at::Device> devices{device};
  const auto localCommsKey = getDeviceIds(devices);
  auto diclComms =
      getDICLComms(localCommsKey, devices, this->rank_, OpType::UNKNOWN);
  return diclComms[0]->getName();
}

c10::intrusive_ptr<Work> ProcessGroupDICL::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs, const AllgatherOptions& opts) {
  checkDeviceTensors(inputs);
  // output = input * ranks, no inplace. every ranks use both in&out.
  auto outputFlattened =
      flatten_for_scatter_gather(outputs, inputs, this->size_);

  auto work = collective(
      inputs, outputFlattened,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclAllgather", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclAllgather", stream.rawstream(),
                                      static_cast<int>(stream.id()));

        return devproxy::diclAllGather(input.data_ptr(), output.data_ptr(),
                                       static_cast<size_t>(input.numel()),
                                       input.scalar_type(), comm,
                                       stream.rawstream());
      },
      [&](std::vector<std::shared_ptr<DICLComm>>& diclComms) {},
      [&](std::vector<std::shared_ptr<DICLComm>>& diclComms) {
        // Copy the flattened output tensors to the outputs.
        for (size_t i = 0; i < outputs.size(); ++i) {
          // warnning & todo:: copy in comm stream,
          // record dest tensor outputs, because src tensor outputFlattened
          // already recorded in collective.
          copyInCommStream<true>(diclComms[i], outputs[i], outputFlattened[i],
                                 static_cast<int>(outputs[i].size()));
          // copyInCurrentStream(diclComms[i], outputs[i], outputFlattened[i]);
        }
      },
      OpType::ALLGATHER);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupDICL::_allgather_base(
    at::Tensor& output, at::Tensor& input, const AllgatherOptions& opts) {
  // output = input * ranks.
  TORCH_CHECK(input.dtype() == output.dtype(),
              "output tensor must have the same type as input tensor");
  TORCH_CHECK(
      input.numel() * this->size_ == output.numel(),
      "output tensor size must be equal to world_size times input tensor size");

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input};
  auto outputs = std::vector<at::Tensor>{output};

  return collective(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclAllgather_base",
                        std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclAllgather_base", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclAllGather(input.data_ptr(), output.data_ptr(),
                                       static_cast<size_t>(input.numel()),
                                       input.scalar_type(), comm,
                                       stream.rawstream());
      },
      OpType::_ALLGATHER_BASE);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::_reduce_scatter_base(
    at::Tensor& output, at::Tensor& input, const ReduceScatterOptions& opts) {
  // input = output * ranks, no inplace, output = reduced(input)[rank]

  TORCH_CHECK(input.dtype() == output.dtype(),
              "output tensor must have the same type as input tensor");
  TORCH_CHECK(
      input.numel() == this->size_ * output.numel(),
      "input tensor must be the same size as output size times world size")

  auto inputs = std::vector<at::Tensor>{input};
  auto outputs = std::vector<at::Tensor>{output};

  return collective(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclReduceScatter_base",
                        std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclReduceScatter_base",
                                      stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclReduceScatter(input.data_ptr(), output.data_ptr(),
                                           static_cast<size_t>(output.numel()),
                                           input.scalar_type(), opts.reduceOp,
                                           comm, stream.rawstream());
      },
      OpType::_REDUCE_SCATTER_BASE);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs, const ScatterOptions& opts) {
  // input = output * ranks, no inplace, output = input[rank] +
  static const auto raise_invalid_arg_func =
      getInvalidArgumentFunc("ProcessGroupDICL::scatter: ");
  int numRanks = getSize();
  int curRank = getRank();
  int rootRank = static_cast<int>(opts.rootRank);
  c10d::assertRootRank(raise_invalid_arg_func, rootRank, numRanks);
  checkDeviceTensors(outputs);
  c10d::assertSingleElementOutput(raise_invalid_arg_func, outputs);
  auto output = outputs.back();
  std::vector<at::Tensor> inputTensors;

  if (curRank == rootRank) {
    checkGatherScatterRootRank(inputs, output, numRanks,
                               raise_invalid_arg_func);
    inputTensors = inputs[0];
  } else {
    if (!inputs.empty()) {
      raise_invalid_arg_func("requires empty input on non-root");
    }
    inputTensors = {};
    inputTensors.emplace_back();
  }

  // NOLINTNEXTLINE(readability-suspicious-call-argument)
  return collective(
      outputs, inputTensors,
      [&](at::Tensor& /* unused */, at::Tensor& /* unused */, diclComm_t comm,
          DIPUStream& stream) {
        std::vector<void*> input_ptr_vec = {};
        if (curRank == rootRank) {
          input_ptr_vec.reserve(inputTensors.size());
          for (auto& inputTensor : inputTensors) {
            input_ptr_vec.push_back(inputTensor.data_ptr());
            dipu::recordStream(inputTensor, stream);
          }
        }

        RECORD_FUNCTION(
            "DiclScatter",
            std::vector<c10::IValue>(inputTensors.begin(), inputTensors.end()));
        profile::RecordBlockCreator _("DiclScatter", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        // since param `sendbuf` is only used in root rank process in
        // diclScatter, we can pass input_ptr_vec.data() to it
        return devproxy::diclScatter(input_ptr_vec.data(), output.data_ptr(),
                                     static_cast<size_t>(output.numel()),
                                     output.scalar_type(), rootRank, curRank,
                                     numRanks, comm, stream.rawstream());
      },
      OpType::SCATTER);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  // input = output * ranks, no inplace, output = reduced(input)[rank]
  checkDeviceTensors(outputs);
  auto inputFlattened =
      flatten_for_scatter_gather(inputs, outputs, this->size_);
  checkDeviceTensors(inputFlattened);

  auto work = collective(
      inputFlattened, outputs,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclReduceScatter", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclReduceScatter", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclReduceScatter(input.data_ptr(), output.data_ptr(),
                                           static_cast<size_t>(output.numel()),
                                           input.scalar_type(), opts.reduceOp,
                                           comm, stream.rawstream());
      },
      [&](std::vector<std::shared_ptr<DICLComm>>& diclComms) {
        // Copy the inputs[i].size nums raw tensor intto flattened
        for (size_t i = 0; i < inputs.size(); ++i) {
          // record src tensor inputs, because dest tensor inputFlattened
          // already recorded in collective
          copyInCommStream<false>(diclComms[i], inputFlattened[i], inputs[i],
                                  static_cast<int>(inputs[0].size()));
        }
      },
      [&](std::vector<std::shared_ptr<DICLComm>>& diclComms) {},
      OpType::REDUCE_SCATTER);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupDICL::alltoall_base(
    at::Tensor& outputTensor, at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts /* unused */) {
  check_device_single_tensor(outputTensor, true);
  check_device_single_tensor(inputTensor, true);
  TORCH_CHECK(outputTensor.scalar_type() == inputTensor.scalar_type(),
              "Tensors must have identical data type")

  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    TORCH_CHECK(outputTensor.numel() == inputTensor.numel(),
                "Tensors must have identical number of elements");
    TORCH_CHECK(outputTensor.size(0) == inputTensor.size(0),
                "Tensors must have identical size in dim 0");
    TORCH_CHECK(outputTensor.size(0) % size_ == 0,
                "Tensor's dim 0 does not divide equally across group size");

    auto outputs = std::vector<at::Tensor>{outputTensor};
    auto inputs = std::vector<at::Tensor>{inputTensor};
    return collective(
        inputs, outputs,
        [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
            DIPUStream& stream) {
          RECORD_FUNCTION("DiclAlltoAllEqualSplit",
                          std::vector<c10::IValue>({input}));
          profile::RecordBlockCreator _("DiclAlltoAllEqualSplit",
                                        stream.rawstream(),
                                        static_cast<int>(stream.id()));
          return devproxy::diclAllToAllEqualSplit(
              input.data_ptr(), output.data_ptr(), outputTensor.numel() / size_,
              output.scalar_type(), comm, stream.rawstream(), rank_, size_);
        },
        OpType::ALLTOALL_BASE);
  }

  c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
  c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
  auto outputs = std::vector<at::Tensor>{outputTensor};
  auto inputs = std::vector<at::Tensor>{inputTensor};
  return collective(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        std::vector<size_t> outputCounts(size_);
        std::vector<size_t> inputCounts(size_);
        std::vector<size_t> outputDisplacements(size_);
        std::vector<size_t> inputDisplacements(size_);
        c10d::computeLengthsAndOffsets(outputSplitSizes, output, &outputCounts,
                                       &outputDisplacements);
        c10d::computeLengthsAndOffsets(inputSplitSizes, input, &inputCounts,
                                       &inputDisplacements);
        RECORD_FUNCTION("DiclAlltoAllUnequalSplit",
                        std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclAlltoAllUnequalSplit",
                                      stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclAllToAllUnequalSplit(
            input.data_ptr(), inputCounts.data(), inputDisplacements.data(),
            output.data_ptr(), outputCounts.data(), outputDisplacements.data(),
            output.scalar_type(), comm, stream.rawstream(), rank_, size_);
      },
      OpType::ALLTOALL_BASE);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  size_t numTensors = outputTensors.size();
  TORCH_CHECK(numTensors == inputTensors.size(),
              "Tensor lists must have identical length")
  c10::Device device = outputTensors[0].device();
  at::ScalarType dataType = outputTensors[0].scalar_type();
  for (const auto i : c10::irange(numTensors)) {
    check_device_single_tensor(outputTensors[i], true);
    check_device_single_tensor(inputTensors[i], true);
    TORCH_CHECK(device == outputTensors[i].device() &&
                    device == inputTensors[i].device(),
                "Tensors must be on the same device")
    TORCH_CHECK(dataType == outputTensors[i].scalar_type() &&
                    dataType == inputTensors[i].scalar_type(),
                "Tensors must have identical data type")
  }

  // TODO(jfxu-st): For CUDA, use NCCL Group Calls for higher performance
  // Ref:
  // https://github.com/pytorch/pytorch/blob/f2d7f235a684c593f5a1ff2ca0b47b47274bfe85/torch/csrc/cuda/nccl.cpp#L916-L941

  // TODO(jfxu-st): For the vendors that don't implement
  // devapis::diclAllToAllUnequalSplit, including CUDA, we need a more
  // performant fallback without using a flattened tensor for relay

  std::vector<int64_t> outputSplitSizes(numTensors);
  std::vector<int64_t> inputSplitSizes(numTensors);
  int64_t outputFlattenedTensorSize = 0;
  for (const auto i : c10::irange(numTensors)) {
    outputSplitSizes[i] = outputTensors[i].numel();
    inputSplitSizes[i] = inputTensors[i].numel();
    outputFlattenedTensorSize += outputTensors[i].numel();
  }
  at::Tensor outputFlattenedTensor = native::nodispatch::empty(
      {outputFlattenedTensorSize},
      at::TensorOptions().device(dipu::DIPU_DEVICE_TYPE).dtype(dataType));
  at::Tensor inputFlattenedTensor = at::cat(inputTensors);

  auto outputs = std::vector<at::Tensor>{outputFlattenedTensor};
  auto inputs = std::vector<at::Tensor>{inputFlattenedTensor};
  return collective(
      inputs, outputs,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        std::vector<size_t> outputCounts(size_);
        std::vector<size_t> inputCounts(size_);
        std::vector<size_t> outputDisplacements(size_);
        std::vector<size_t> inputDisplacements(size_);
        c10d::computeLengthsAndOffsets(outputSplitSizes, output, &outputCounts,
                                       &outputDisplacements);
        c10d::computeLengthsAndOffsets(inputSplitSizes, input, &inputCounts,
                                       &inputDisplacements);
        RECORD_FUNCTION("DiclAlltoAllUnequalSplit",
                        std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("DiclAlltoAllUnequalSplit",
                                      stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclAllToAllUnequalSplit(
            input.data_ptr(), inputCounts.data(), inputDisplacements.data(),
            output.data_ptr(), outputCounts.data(), outputDisplacements.data(),
            output.scalar_type(), comm, stream.rawstream(), rank_, size_);
      },
      [&](std::vector<std::shared_ptr<DICLComm>>&) {},
      [&](std::vector<std::shared_ptr<DICLComm>>& comms) {
        DIPUStreamGuard _(comms[0]->diclStream_.unwrap());
        size_t offset = 0;
        for (const auto i : c10::irange(numTensors)) {
          outputTensors[i].copy_(
              outputs[0].slice(0, offset, offset + outputSplitSizes[i]));
          offset += outputSplitSizes[i];
        }
      },
      OpType::ALLTOALL);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  checkDeviceTensors(tensors);
  auto p2pPair = mapPGRank2P2P(rank_, dstRank);
  return pointToPoint(
      tensors, tensors, dstRank,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("diclSend", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("diclSend", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclSend(
            input.data_ptr(), static_cast<size_t>(input.numel()),
            input.scalar_type(), p2pPair.second, comm, stream.rawstream());
      },
      [](std::vector<std::shared_ptr<DICLComm>>&) {},
      [](std::vector<std::shared_ptr<DICLComm>>&) {}, OpType::SEND);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  checkDeviceTensors(tensors);
  auto p2pPair = mapPGRank2P2P(rank_, srcRank);
  return pointToPoint(
      tensors, tensors, srcRank,
      [&](at::Tensor& input, at::Tensor& output, diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("diclRecv", std::vector<c10::IValue>({input}));
        profile::RecordBlockCreator _("diclRecv", stream.rawstream(),
                                      static_cast<int>(stream.id()));
        return devproxy::diclRecv(
            input.data_ptr(), static_cast<size_t>(input.numel()),
            input.scalar_type(), p2pPair.second, comm, stream.rawstream());
      },
      [](std::vector<std::shared_ptr<DICLComm>>&) {},
      [](std::vector<std::shared_ptr<DICLComm>>&) {}, OpType::RECV);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::barrier(const BarrierOptions& opts) {
  std::vector<at::Device> devices;
  if (usedDeviceIdxs_.empty()) {
    auto numDIPUs = devproxy::getDeviceCount();
    int16_t deviceIdx =
        static_cast<int16_t>(rank_ % std::max(static_cast<int>(numDIPUs), 1));
    devices.emplace_back(dipu::DIPU_DEVICE_TYPE,
                         static_cast<c10::DeviceIndex>(deviceIdx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.emplace_back(dipu::DIPU_DEVICE_TYPE,
                           static_cast<c10::DeviceIndex>(usedDeviceIdx));
    }
  }

  std::vector<at::Tensor> barrierTensors{};
  barrierTensors.reserve(devices.size());

  OptionalDIPUGuard dipuGuard;
  for (auto& device : devices) {
    dipuGuard.reset_device(device);
    barrierTensors.push_back(at::empty(
        {1},
        at::TensorOptions().device(dipu::DIPU_DEVICE_TYPE).dtype(at::kFloat)));
  }

  auto work = allreduce(barrierTensors, AllreduceOptions());
  auto diclWork = dynamic_cast<ProcessGroupDICL::WorkDICL*>(work.get());
  diclWork->barrier_ = true;

  return work;
}

c10::intrusive_ptr<ProcessGroupDICL> createProcessGroupDICL(
    const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
    const std::chrono::milliseconds& timeout) {
  auto options = c10::make_intrusive<ProcessGroupDICL::Options>();
  options->timeout = timeout;
  return c10::make_intrusive<ProcessGroupDICL>(store, rank, size);
}

}  // namespace dipu
