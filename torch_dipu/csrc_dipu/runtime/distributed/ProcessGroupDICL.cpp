// Copyright (c) 2023, DeepLink.
#include <ATen/record_function.h>
#include <torch/torch.h>

#include <csrc_dipu/runtime/core/DIPUGuard.h>
#include <csrc_dipu/utils/helpfunc.hpp>
#include "./ProcessGroupDICL.h"
namespace dipu {
namespace {

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
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

static std::string getKeySendRecv(int my_rank, int peer) {
  int low_rank = my_rank < peer ? my_rank : peer;
  int high_rank = my_rank < peer ? peer : my_rank;
  std::string send_recv_pair =
      std::to_string(low_rank) + ":" + std::to_string(high_rank);
  return send_recv_pair;
}

// Get the list of devices from list of tensors
static std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

static void syncStreams(std::vector<std::shared_ptr<DICLComm>>& comms) {
  for (size_t i = 0; i < comms.size(); ++i) {
    comms[i]->preSyncStream();
  }
}

}  // anonymous namespace

// start WorkDICL

// ProcessGroupDICL::WorkDICL::~WorkDICL() {}

// currently DICL do not support error check
bool ProcessGroupDICL::WorkDICL::isCompleted() { return finishedDICLExecutionInternal(); }

// currently DICL do not support error check
bool ProcessGroupDICL::WorkDICL::isSuccess() const { return finishedDICLExecutionInternal(); }

bool ProcessGroupDICL::WorkDICL::finishedDICLExecutionInternal() const {
  for (auto& workEvent : workEvents_) {
    if (!workEvent.query()) {
      return false;
    }
  }
  return true;
}

// record post work event on communicator stream
void ProcessGroupDICL::WorkDICL::record() {
  for (auto i = 0; i < workEvents_.size(); i++) {
    workEvents_[i].record(diclComms_[i]->diclStream_);
  }
}

void ProcessGroupDICL::WorkDICL::synchronize() {
  for (auto i = 0; i < workEvents_.size(); i++) {
    auto currentStream = dipu::getCurrentDIPUStream(diclComms_[i]->device_.index());
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

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupDICL::WorkDICL::
    getFuture() {
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
          throw std::runtime_error(
              "Invalid value for environment variable: " +
              std::string(DICL_BLOCKING_WAIT));
        }
      }
    } catch (std::exception& e) {
      throw std::runtime_error(
          "Invalid value for environment variable: " +
          std::string(DICL_BLOCKING_WAIT));
    }
  }

ProcessGroupDICL::~ProcessGroupDICL() {}

void ProcessGroupDICL::broadcastUniqueID(commUniqueId* uniqueId, bool isSingleP2POp,
      const std::string& p2pKey, int p2pRank) {
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

  std::string storeKey;
  if (!isSingleP2POp) {
    storeKey = std::to_string(diclCommCounter_++);
  } else {
    storeKey  = p2pKey;
  }
  if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(uniqueId),
        reinterpret_cast<uint8_t*>(uniqueId) + devapis::DICL_UNIQUE_ID_BYTES_SIZE);
    store_->set(storeKey, vec);
  } else {
    auto vec = store_->get(storeKey);
    if (vec.size() !=  devapis::DICL_UNIQUE_ID_BYTES_SIZE) {
      throw std::runtime_error(
          "Unexpected DICL unique ID length received "
          "from the store");
    }
    std::memcpy(uniqueId, vec.data(), vec.size());
  }
}

std::vector<std::shared_ptr<DICLComm>>& ProcessGroupDICL::getDICLComms(const std::string& devicesKey,
      const std::vector<at::Device>& devices, OpType opType, int p2pRank, bool isSendRecvSelf) {
 // Sanity check
  if (devicesKey.empty()) {
    throw std::runtime_error(
        "Not able to create/get the DICL Communicator since "
        "the DIPU devices are not known");
  }
  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }
  {
    std::lock_guard<std::mutex> lock(devDICLCommMapLock_);
    if (devDICLCommsMap_.find(devicesKey) != devDICLCommsMap_.end()) {
      // Reuse the cached communicator if there is one.
      return devDICLCommsMap_[devicesKey];
    }
  }
  // not cached, create a new entry
  std::vector<std::shared_ptr<DICLComm>> diclComms;
  auto devSize = devices.size();
  diclComms.resize(devSize);

  commUniqueId diclID;

  bool singleP2POp = isP2POp(opType, false);
  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
    devproxy::diclGetUniqueId(&diclID);
  }

  broadcastUniqueID(&diclID, singleP2POp, devicesKey, p2pRank);

  OptionalDIPUGuard dipuGuard;

  for (int i=0; i < devSize; i++) {
    int numRanks = getSize();
    int rank = getRank() * devSize + i;
    dipuGuard.reset_device(devices[i]);

    // need use pool stream, not current stream. fix stream guard err.
    auto commStream = getDIPUStreamFromPool(devices[i].index());
    // auto commStream = getCurrentDIPUStream(devices[i].index());

    diclComms[i] = DICLComm::create(numRanks, rank, diclID, commStream);
  }

  // Hold the lock before modifying the cache.
  std::lock_guard<std::mutex> lock(devDICLCommMapLock_);
  // Move the DICL resource to cache
  devDICLCommsMap_.emplace(devicesKey, std::move(diclComms));
  return devDICLCommsMap_[devicesKey];
}

namespace {

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(std::vector<std::vector<at::Tensor>>& tensor_lists,
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
}  // annoy namespace


// Check that all `tensors', different device may need extend this func to do device specific check
void ProcessGroupDICL::checkDeviceTensors(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(devproxy::getDeviceCount())) {
    throw std::runtime_error(
        "Tensor list mustn't be larger than the number of available DIPUs");
  }
  for (auto tensor: tensors) {
    if (!dipu::isDeviceTensor(tensor) || !tensor.is_non_overlapping_and_dense()) {
      throw std::runtime_error("Tensors must be DIPU and non-overlapping and dense");
    }
  }
}

// std::function< diclResult_t(at::Tensor&, at::Tensor&, DiclComm, DIPUStream&) >
// enhance: need change template params to lamada, make collective() func overridable by sub class
template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupDICL::collective(
    std::vector<at::Tensor>& inputs, std::vector<at::Tensor>& outputs, Fn fn,
    PreProcess pre, PostProcess post, OpType opType) {
  const auto devices = getDeviceList(inputs);

  TORCH_CHECK(devices.size() == 1,
    "dipu support one device per process only, multidevices is not fully tested ", 
    " you can comment out this assert if you try to test,");
  const auto key = getKeyFromDevices(devices);
  auto diclComms = getDICLComms(key, devices, opType);

  // First let DICL streams wait for input tensors allocation streams
  syncStreams(diclComms);
  auto work = c10::make_intrusive<ProcessGroupDICL::WorkDICL>(diclComms, blockingWait_, opTimeout_);

  OptionalDIPUGuard dipuGuard;
  pre(diclComms);

  for (size_t i = 0; i < inputs.size(); ++i) {
    dipuGuard.reset_device(devices[i]);

    // need add adapter to handle int64/double! camb not support double
    fn(inputs[i], outputs[i], diclComms[i]->rawComm(), diclComms[i]->diclStream_);

    // todo:: add recordStream after cacheAllocator ready
    // DIPUCachingAllocator::recordStream(outputs[i].storage().data_ptr(), stream);

    // mock comm with just copy, used in standalone test.
    // DIPUStreamGuard guard(diclComms[i]->diclStream_.unwrap());
    // outputs[i].copy_(inputs[i], false);  
  }

  post(diclComms);
  work->record();

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);
  // todo:: dipu need support multistream guard & remove work->workEvents_(future already has events ).
  {
    DIPUStreamGuard streamGuard(diclComms[0]->diclStream_);

    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }
  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupDICL::collective(std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs, Fn fn, OpType opType) {
  return collective(inputs, outputs, fn, [](std::vector<std::shared_ptr<DICLComm>>&) {},
                    [](std::vector<std::shared_ptr<DICLComm>>&) {}, opType);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  checkDeviceTensors(tensors);
  return collective(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          diclComm_t comm,
          DIPUStream& stream) {
        RECORD_FUNCTION("DiclAllreduce", std::vector<c10::IValue>({input}));
        return devproxy::diclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            input.scalar_type(),
            opts.reduceOp,
            comm,
            stream.rawstream());
      },
      OpType::ALLREDUCE);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  checkDeviceTensors(tensors);
  return collective(
    tensors,
    tensors,
    [&](at::Tensor& input,
        at::Tensor& output,
        diclComm_t comm,
        DIPUStream& stream) {
      RECORD_FUNCTION("DiclBroadcast", std::vector<c10::IValue>({input}));
      const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
      return devproxy::diclBroadcast(
          input.data_ptr(),
          input.data_ptr(),
          (size_t)input.numel(),
          input.scalar_type(),
          root,
          comm,
          stream.rawstream());
    },
    OpType::BROADCAST);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  throw std::runtime_error("ProcessGroupDICL does not support reduce now");
}

c10::intrusive_ptr<Work> ProcessGroupDICL::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors, const AllgatherOptions& opts) {
  checkDeviceTensors(input_tensors);

  auto outputFlattened =
      flatten_for_scatter_gather(output_tensors, input_tensors, this->size_);

  auto work = collective(input_tensors, outputFlattened,
    [&](at::Tensor& input,
        at::Tensor& output,
        diclComm_t comm,
        DIPUStream& stream) {
      RECORD_FUNCTION("DiclAllgather", std::vector<c10::IValue>({input}));

      return devproxy::diclAllGather(
          input.data_ptr(),
          output.data_ptr(),
          (size_t)input.numel(),
          input.scalar_type(),
          comm,
          stream.rawstream());
    },
    [&](std::vector<std::shared_ptr<DICLComm>>& diclComms) {},
    [&](std::vector<std::shared_ptr<DICLComm>>& diclComms) {
      // Copy the flattened output tensors to the outputs.
      for (size_t i = 0; i < output_tensors.size(); ++i) {
        DIPUStreamGuard guard(diclComms[i]->diclStream_.unwrap());
        for (size_t j = 0; j < output_tensors[0].size(); ++j) {
          output_tensors[i][j].copy_(outputFlattened[i][j], false);
          //todo:: add recordStream after cacheAllocator ready
          // DIPUCachingAllocator::recordStream(
          //     output_tensors[i][j].storage().data_ptr(), diclComms[i]->diclStream_);
        }
      }
    }, 
    OpType::ALLGATHER);
    // std::cout << outputFlattened[0] << std::endl;
    // std::cout << output_tensors[0][0] << std::endl;
    return work;
}

c10::intrusive_ptr<Work> ProcessGroupDICL::_allgather_base(
    at::Tensor& output_tensor, at::Tensor& input_tensor, const AllgatherOptions& opts) {
  
  if (input_tensor.dtype() != output_tensor.dtype()) {
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (input_tensor.numel() * this->size_ != output_tensor.numel()) {
    TORCH_CHECK(false,
        "output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor>{input_tensor};
  auto outputs = std::vector<at::Tensor>{output_tensor};
  checkDeviceTensors(inputs);
  checkDeviceTensors(outputs);

  return collective(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          diclComm_t comm,
          DIPUStream& stream) {
        return devproxy::diclAllGather(
          input.data_ptr(),
          output.data_ptr(),
          (size_t)input.numel(),
          input.scalar_type(),
          comm,
          stream.rawstream());
      },
      OpType::_ALLGATHER_BASE);
}


c10::intrusive_ptr<Work> ProcessGroupDICL::send(
    std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  checkDeviceTensors(tensors);
  return collective(
    tensors,
    tensors,
    [&](at::Tensor& input,
        at::Tensor& output,
        diclComm_t comm,
        DIPUStream& stream) {
      RECORD_FUNCTION("diclSend", std::vector<c10::IValue>({input}));
      return devproxy::diclSend(
          input.data_ptr(),
          (size_t)input.numel(),
          input.scalar_type(),
          dstRank,
          comm,
          stream.rawstream());
    },
    OpType::SEND);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::recv(
    std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  checkDeviceTensors(tensors);
  return collective(
    tensors,
    tensors,
    [&](at::Tensor& input,
        at::Tensor& output,
        diclComm_t comm,
        DIPUStream& stream) {
      RECORD_FUNCTION("diclRecv", std::vector<c10::IValue>({input}));
      return devproxy::diclRecv(
          input.data_ptr(),
          (size_t)input.numel(),
          input.scalar_type(),
          srcRank,
          comm,
          stream.rawstream());
    },
    OpType::RECV);
}

c10::intrusive_ptr<Work> ProcessGroupDICL::barrier(
    const BarrierOptions& opts) {
  std::vector<at::Device> devices;
  if (usedDeviceIdxs_.empty()) {
    auto numDIPUs = devproxy::getDeviceCount();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % std::max(static_cast<int>(numDIPUs), 1));
    devices.push_back(at::Device(dipu::DIPU_DEVICE_TYPE, deviceIdx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.push_back(at::Device(dipu::DIPU_DEVICE_TYPE, usedDeviceIdx));
    }
  }

  std::vector<at::Tensor> barrierTensors;
  barrierTensors.reserve(devices.size());

  OptionalDIPUGuard dipuGuard;
  for (auto& device : devices) {
    dipuGuard.reset_device(device);
    barrierTensors.push_back(at::empty({1},
        at::TensorOptions().device(dipu::DIPU_DEVICE_TYPE).dtype(at::kFloat)));
  }

  auto work = allreduce(barrierTensors);
  auto diclWork = dynamic_cast<ProcessGroupDICL::WorkDICL*>(work.get());
  diclWork->barrier_ = true;

  return work;
}

c10::intrusive_ptr<ProcessGroupDICL> createProcessGroupDICL(
      const c10::intrusive_ptr<::c10d::Store> &store,
      int rank,
      int size,
      const std::chrono::milliseconds& timeout) {
  auto options = c10::make_intrusive<ProcessGroupDICL::Options>();
  options->timeout = timeout;
  return c10::make_intrusive<ProcessGroupDICL>(store, rank, size);
}

}  // namespace c10d
