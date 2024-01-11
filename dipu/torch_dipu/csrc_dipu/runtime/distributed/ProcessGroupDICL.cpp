// Copyright (c) 2023, DeepLink.
#include "ProcessGroupDICL.h"

#include <utility>

#include <ATen/record_function.h>
#include <torch/torch.h>

#include "csrc_dipu/profiler/profiler.h"
#include "csrc_dipu/runtime/core/DIPUGuard.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/utils/helpfunc.hpp"

namespace dipu {

using std::pair;

namespace {

// Get the list of devices from list of tensors, collective comm always use all
// ranks, so no rank prefix required in key.
std::string getDevieceIds(const std::vector<at::Device>& devices) {
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

}  // anonymous namespace

// start WorkDICL

// ProcessGroupDICL::WorkDICL::~WorkDICL() {}

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
// NOLINTNEXTLINE(google-default-arguments)
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
    : c10d::Backend(rank, size), store_(store) {
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
    store_->set(storeKey, vec);
  } else {
    auto vec = store_->get(storeKey);
    if (vec.size() != devapis::DICL_UNIQUE_ID_BYTES_SIZE) {
      throw std::runtime_error(
          "Unexpected DICL unique ID length received "
          "from the store");
    }
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

// Check that all `tensors', different device may need extend this func to do
// device specific check
void ProcessGroupDICL::checkDeviceTensors(
    const std::vector<at::Tensor>& tensors) {
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

  OptionalDIPUGuard dipuGuard;
  pre(diclComms);

  for (size_t i = 0; i < inputs.size(); ++i) {
    dipuGuard.reset_device(diclComms[i]->device_);

    // need add adapter to handle int64/double! camb not support double
    fn(inputs[i], outputs[i], diclComms[i]->rawComm(),
       diclComms[i]->diclStream_);

    dipu::recordStream(inputs[i], diclComms[i]->diclStream_);
    if (inputs[i].storage().data_ptr().get() !=
        outputs[i].storage().data_ptr().get()) {
      dipu::recordStream(outputs[i], diclComms[i]->diclStream_);
    }

    // mock comm with just copy, used in standalone test.
    // DIPUStreamGuard guard(diclComms[i]->diclStream_.unwrap());
    // outputs[i].copy_(inputs[i], false);
  }

  post(diclComms);
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

  const auto localCommsKey = getDevieceIds(devices);

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

// NOLINTNEXTLINE(google-default-arguments)
c10::intrusive_ptr<Work> ProcessGroupDICL::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  // inplace in = out, every rank use both in&out.
  checkDeviceTensors(tensors);
  return collective(
      tensors, tensors,
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
      OpType::ALLREDUCE);
}

// NOLINTNEXTLINE(google-default-arguments)
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

// NOLINTNEXTLINE(google-default-arguments)
c10::intrusive_ptr<Work> ProcessGroupDICL::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  // inplace in = out, only rootRank use out.
  checkDeviceTensors(tensors);

  auto tensor = tensors.back();
  int dev_in_group = 0;
  return collective(
      tensors, tensors,
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
      OpType::REDUCE);
}

// NOLINTNEXTLINE(google-default-arguments)
c10::intrusive_ptr<Work> ProcessGroupDICL::gather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs, const GatherOptions& opts) {
  TORCH_CHECK(false, "ProcessGroupDICL does not support gather now");
}

// NOLINTNEXTLINE(google-default-arguments)
c10::intrusive_ptr<Work> ProcessGroupDICL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors, const AllgatherOptions& opts) {
  checkDeviceTensors(inputTensors);
  // output = input * ranks, no inplace. every ranks use both in&out.
  auto outputFlattened =
      flatten_for_scatter_gather(outputTensors, inputTensors, this->size_);

  auto work = collective(
      inputTensors, outputFlattened,
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
        for (size_t i = 0; i < outputTensors.size(); ++i) {
          // warnning & todo:: copy in comm stream,
          // record dest tensor outputs, because src tensor outputFlattened
          // already recorded in collective.
          copyInCommStream<true>(diclComms[i], outputTensors[i],
                                 outputFlattened[i],
                                 static_cast<int>(outputTensors[i].size()));
          // copyInCurrentStream(diclComms[i], outputs[i], outputFlattened[i]);
        }
      },
      OpType::ALLGATHER);
  return work;
}

// NOLINTNEXTLINE(google-default-arguments)
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

// NOLINTNEXTLINE(google-default-arguments)
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

// NOLINTNEXTLINE(google-default-arguments)
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

// NOLINTNEXTLINE(google-default-arguments)
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
