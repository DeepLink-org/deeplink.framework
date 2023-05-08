/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "distributed/process_group_cncl.hpp"

#include <map>

#ifdef TEST_COVERAGE
extern "C" void __gcov_flush();
#endif

namespace c10d {

namespace {

// CNCL op mapping
std::map<ReduceOp, cnclReduceOp_t> cncl_op = {
    {ReduceOp::MIN, cnclMin},
    {ReduceOp::MAX, cnclMax},
    {ReduceOp::SUM, cnclSum},
    {ReduceOp::PRODUCT, cnclProd},
};

// CNCL type typing
std::map<at::ScalarType, cnclDataType_t> cncl_data_type = {
    {at::kChar, cnclInt8}, {at::kByte, cnclUint8}, {at::kFloat, cnclFloat},
    {at::kInt, cnclInt32}, {at::kLong, cnclInt32}, {at::kHalf, cnclHalf},
    {at::kDouble, cnclFloat}
};

// Helper function that gets the data type and issues error if not supported
cnclDataType_t getCnclDataType(at::ScalarType type) {
  try {
    return cncl_data_type.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for CNCL process group");
  }
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  return std::to_string(devices[0].index());
}

std::string getKeySendRecv(int my_rank, int peer) {
  int low_rank = my_rank < peer ? my_rank : peer;
  int high_rank = my_rank < peer ? peer : my_rank;
  std::string send_recv_pair =
      std::to_string(low_rank) + ":" + std::to_string(high_rank);
  return send_recv_pair;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

void syncQueues(const at::Device& device, torch_mlu::Notifier& cncl_notifier,
                torch_mlu::Queue& cncl_queue) {
  auto current_queue = torch_mlu::getCurrentQueue(device.index());
  cncl_notifier.place(current_queue);
  cncl_notifier.wait(cncl_queue);
  if (torch_mlu::PythonInterface::getAsyncMode() == false) {
    current_queue.synchronize();
  }
}

}  // namespace

ProcessGroupCNCL::WorkCNCL::WorkCNCL(const std::vector<at::Device>& devices)
    : devices_(devices) {}

ProcessGroupCNCL::WorkCNCL::~WorkCNCL() {}

// currently MLU do not support CNCL error check
bool ProcessGroupCNCL::WorkCNCL::isCompleted() { return finishedMLUExecutionInternal(); }

// currently MLU do not support CNCL error check
bool ProcessGroupCNCL::WorkCNCL::isSuccess() const { return finishedMLUExecutionInternal(); }

bool ProcessGroupCNCL::WorkCNCL::finishedMLUExecutionInternal() const {
  // Checking the work's corresponding MLU notifier's status
  return notifier_.query();
}

// Waiting on the work's corresponding CNRT events
void ProcessGroupCNCL::WorkCNCL::synchronize() {
  auto current_queue = torch_mlu::getCurrentQueue();
  notifier_.wait(current_queue);

  if (blockingWait_) {
    current_queue.synchronize();
  }

  if (!barrier_tensors_.empty()) {
    torch_mlu::mlu::MLUGuard guard(devices_[0]);
    TORCH_CNRT_CHECK(cnrtSyncDevice());
  }

  return;
}

// Same as calling synchronize().
bool ProcessGroupCNCL::WorkCNCL::wait(std::chrono::milliseconds timeout) {
  synchronize();
  return true;
}

ProcessGroupCNCL::ProcessGroupCNCL(const c10::intrusive_ptr<Store>& store,
                                   int rank, int size)
  : ProcessGroup(rank, size), store_(store) {
    char* blockingWait = getenv(CNCL_BLOCKING_WAIT);
    try {
      if (blockingWait != nullptr) {
        auto val = std::stoi(blockingWait);
        if (val == 1) {
          // Make wait() and synchronize() a blocking call.
          blockingWait_ = true;
        } else if (val != 0) {
          throw std::runtime_error(
              "Invalid value for environment variable: " +
              std::string(CNCL_BLOCKING_WAIT));
        }
      }
    } catch (std::exception& e) {
      throw std::runtime_error(
          "Invalid value for environment variable: " +
          std::string(CNCL_BLOCKING_WAIT));
    }
  }

ProcessGroupCNCL::~ProcessGroupCNCL() {
  // TODO(zhanchendi): cnclDestroyComms do not wait for uncompleted operations
  // before destroying the communicator as ncclCommDestroy, and in the future,
  // CNCL may support cnclAbortComm normally
  auto is_completed = dev_cncl_comm_map_.empty() || cncl_queue_.query();
  if (!is_completed) {
    CNLOG(WARNING) << "In async mode, CNCL task may be uncompleted "
                   << "when process group is deconstructed for rank " << rank_
                   << ". We will wait for work to complete forever.";
    cncl_queue_.synchronize();
  }
// gcov can not save the coverage data of the code run by subprocess,
// so we flush the coverge data manually
#ifdef TEST_COVERAGE
  __gcov_flush();
#endif
}

void ProcessGroupCNCL::broadcastCNCLCliqueID(
    cnclCliqueId* cncl_id,
    const bool is_p2p_op = false,
    const std::string& p2p_key = "",
    const int p2p_rank = 0) {
  // For collective operations:
  // For every CNCL communicator that we create we need to broadcast
  // a unique ID from rank 0 to all other ranks. This broadcast is
  // done by rank 0 setting a key in the store and all other ranks
  // retrieving the contents of that key. A single process group
  // may create multiple CNCL communicators, so we use a sequence
  // number to differentiate between them.
  // For point-to-point operations:
  // The sequence number will only be increased on 2 out of all the
  // processes in a Process Group. So all following collective
  // operations will see different sequence numbers which will cause
  // runtime errors. To avoid that, use the src:target pair instead
  // of sequence number for p2p communications.

  std::string store_key;
  if (!is_p2p_op) {
    store_key = std::to_string(cncl_comm_counter_++);
  } else {
    store_key  = p2p_key;
  }
  if (rank_ == 0 || (is_p2p_op && p2p_rank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(cncl_id),
        reinterpret_cast<uint8_t*>(cncl_id) + CNCL_CLIQUE_ID_BYTES_SIZE);
    store_->set(store_key, vec);
  } else {
    auto vec = store_->get(store_key);
    if (vec.size() != CNCL_CLIQUE_ID_BYTES_SIZE) {
      throw std::runtime_error(
          "Unexpected CNCL clique ID length received "
          "from the store");
    }
    std::memcpy(cncl_id, vec.data(), vec.size());
  }
}

std::shared_ptr<CNCLComms>& ProcessGroupCNCL::getCNCLComms(
    const std::string& devices_key,
    const std::vector<at::Device>& devices,
    const bool is_p2p_op = false,
    const int p2p_rank = 0) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
        "Not able to create/get the CNCL Communicator since "
        "the MLU devices are not known");
  }

  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
  }

  {
    if (dev_cncl_comm_map_.find(devices_key) != dev_cncl_comm_map_.end()) {
      // Reuse the cached communicator if there is one.
      return dev_cncl_comm_map_[devices_key];
    }
  }
  // CNCL communicator not cached, create a new entry
  std::shared_ptr<CNCLComms> cncl_comms;

  // Create the unique CNCL ID and broadcast it
  cnclCliqueId clique_id;

  // For point-to-point communication, lower rank of the two will get unique id.
  if (rank_ == 0 || (is_p2p_op && p2p_rank == 0)) {
    C10D_CNCL_CHECK(cnclGetCliqueId(&clique_id));
  }

  // Broadcast so that each process can have a unique CNCL ID
  broadcastCNCLCliqueID(&clique_id, is_p2p_op, devices_key, p2p_rank);

  // MLU local size and global size
  int num_size = size_;
  int num_comms = 1;
  // MLU local rank an global rank
  int rank_id = rank_;
  if (is_p2p_op) {
    num_size = 2;
    rank_id = p2p_rank;
  }

  std::vector<int> dev_list{devices[0].index()};
  std::vector<int> rank_list{rank_id * num_comms};
  // Create the CNCL communicators for each MLU
  cncl_comms = CNCLComms::create(num_comms, &(dev_list[0]), &(rank_list[0]),
                                 num_size, &clique_id);
  cncl_queue_ = torch_mlu::getQueueFromPool(devices[0].index());

  dev_cncl_comm_map_.emplace(devices_key, std::move(cncl_comms));
  return dev_cncl_comm_map_[devices_key];
}

namespace {

// Check that all `tensors' have the same type and shape and are distributed
// across distinct MLUs.
void check_mlu_tensors(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(torch_mlu::device_count())) {
    throw std::runtime_error(
        "Tensor list mustn't be larger than the number of available MLUs");
  }

  const auto& first = tensors.front();
  if (!first.is_mlu() || first.is_sparse()) {
    throw std::runtime_error("Tensors must be MLU and dense");
  }
  if (!first.is_non_overlapping_and_dense()) {
    throw std::runtime_error("Tensors must be non-overlapping and dense");
  }

  if (tensors.size() != 1) {
    throw std::runtime_error(
        "MLU Tensors must be on a single MLU device per process");
  }
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
at::Tensor flatten_tensor_list(std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other, size_t world_size) {
  if (tensor_lists.size() != 1 || other.size() != 1) {
    throw std::runtime_error(
        "MLU Tensors must be on a single MLU device per process");
  }

  auto device = other[0].device();
  for (const auto& t : tensor_lists[0]) {
    if (t.numel() != other[0].numel()) {
      throw std::runtime_error(
          "All tensor operands to scatter/gather must have the same size");
    }
    if (t.device() != device) {
      throw std::runtime_error("Expecting all tensors on the same device");
    }
  }

  std::vector<int64_t> new_size{static_cast<int64_t>(tensor_lists[0].size())};
  new_size.insert(new_size.end(), tensor_lists[0][0].sizes().begin(),
                  tensor_lists[0][0].sizes().end());
  return at::empty(new_size, tensor_lists[0][0].options());
}

}  // namespace

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::allreduce(
    std::vector<at::Tensor>& tensors, const AllreduceOptions& opts) {
  check_mlu_tensors(tensors);
  auto devices = getDeviceList(tensors);
  auto key = getKeyFromDevices(devices);
  auto& cncl_comms = getCNCLComms(key, devices);
  syncQueues(devices[0], cncl_notifier_, cncl_queue_);

  // Work itself will create the CNCL events on all MLUs of tensors
  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  auto input_impl = torch_mlu::getMluTensorImpl(tensors[0]);
  auto input_ptr = input_impl->mlu_data_ptr();

  torch_mlu::recordQueue(tensors[0].storage().data_ptr(), cncl_queue_);
  C10D_CNCL_CHECK(cnclAllReduce(
      input_ptr, input_ptr, tensors[0].numel(),
      getCnclDataType(tensors[0].scalar_type()), cncl_op[opts.reduceOp],
      cncl_comms->getCnclComm(0), cncl_queue_.queue()));
  work->notifier_.place(cncl_queue_);
  work->blockingWait_ = blockingWait_;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::broadcast(
    std::vector<at::Tensor>& tensors, const BroadcastOptions& opts) {
  check_mlu_tensors(tensors);

  auto devices = getDeviceList(tensors);
  auto key = getKeyFromDevices(devices);
  auto& cncl_comms = getCNCLComms(key, devices);
  syncQueues(devices[0], cncl_notifier_, cncl_queue_);

  // Work itself will create the CNCL events on all MLUs of tensors
  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  auto input_impl = torch_mlu::getMluTensorImpl(tensors[0]);
  auto input_ptr = input_impl->mlu_data_ptr();
  int root = opts.rootRank * tensors.size() + opts.rootTensor;
  torch_mlu::recordQueue(tensors[0].storage().data_ptr(), cncl_queue_);
  C10D_CNCL_CHECK(cnclBroadcast(input_ptr, input_ptr, tensors[0].numel(),
                                getCnclDataType(tensors[0].scalar_type()),
                                root, cncl_comms->getCnclComm(0),
                                cncl_queue_.queue()));
  work->notifier_.place(cncl_queue_);
  work->blockingWait_ = blockingWait_;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::reduce(
    std::vector<at::Tensor>& tensors, const ReduceOptions& opts) {
  check_mlu_tensors(tensors);

  auto devices = getDeviceList(tensors);
  auto key = getKeyFromDevices(devices);
  auto& cncl_comms = getCNCLComms(key, devices);
  syncQueues(devices[0], cncl_notifier_, cncl_queue_);

  // Work itself will create the CNCL events on all MLUs of tensors
  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  auto input_impl = torch_mlu::getMluTensorImpl(tensors[0]);
  auto input_ptr = input_impl->mlu_data_ptr();
  int root = opts.rootRank * tensors.size() + opts.rootTensor;
  torch_mlu::recordQueue(tensors[0].storage().data_ptr(), cncl_queue_);
  C10D_CNCL_CHECK(cnclReduce(input_ptr, input_ptr, tensors[0].numel(),
                  getCnclDataType(tensors[0].scalar_type()), cncl_op[opts.reduceOp],
                  root, cncl_comms->getCnclComm(0), cncl_queue_.queue()));
  work->notifier_.place(cncl_queue_);
  work->blockingWait_ = blockingWait_;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors, const AllgatherOptions& opts) {
  check_mlu_tensors(input_tensors);
  auto flattened = flatten_tensor_list(output_tensors, input_tensors, size_);

  auto devices = getDeviceList(input_tensors);
  auto key = getKeyFromDevices(devices);
  auto& cncl_comms = getCNCLComms(key, devices);
  syncQueues(devices[0], cncl_notifier_, cncl_queue_);

  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  auto flattened_impl = torch_mlu::getMluTensorImpl(flattened);
  auto flattened_ptr = flattened_impl->mlu_data_ptr();
  auto input_impl = torch_mlu::getMluTensorImpl(input_tensors[0]);
  auto input_ptr = input_impl->mlu_data_ptr();

  torch_mlu::recordQueue(flattened.storage().data_ptr(), cncl_queue_);
  C10D_CNCL_CHECK(
      cnclAllGather(input_ptr, flattened_ptr, input_tensors[0].numel(),
                    getCnclDataType(input_tensors[0].scalar_type()),
                    cncl_comms->getCnclComm(0), cncl_queue_.queue()));

  torch_mlu::mlu::MLUQueueGuard guard(cncl_queue_);
  for (size_t i = 0; i < output_tensors[0].size(); ++i) {
    torch_mlu::recordQueue(output_tensors[0][i].storage().data_ptr(),
                           cncl_queue_);
    // Temperarily fix for MLU temperary layout rule on chip
    auto flattened_view = at::empty(input_tensors[0].sizes(), input_tensors[0].options());
    flattened_view.set_(flattened.storage(), i * input_tensors[0].numel(),
        output_tensors[0][i].sizes(), output_tensors[0][i].strides());
    output_tensors[0][i].copy_(flattened_view);
    // output_tensors[0][i].copy_(flattened[i]);
  }
  work->notifier_.place(cncl_queue_);
  work->blockingWait_ = blockingWait_;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::_allgather_base(
    at::Tensor& outputBuffer,
    at::Tensor& inputBuffer,
    const AllgatherOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& outputTensorLists,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::barrier(
    const BarrierOptions& opts) {
  std::vector<at::Device> devices;
  if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a CNCL collective being called
    // Here we have to use the best guesses and will use a single MLU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different MLU
    auto num_mlus = torch_mlu::device_count();
    int16_t device_idx = static_cast<int16_t>(rank_ % num_mlus);
    devices.push_back(at::Device(at::DeviceType::MLU, device_idx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.push_back(at::Device(at::DeviceType::MLU, usedDeviceIdx));
    }
  }

  std::vector<at::Tensor> barrier_tensors;
  barrier_tensors.reserve(devices.size());

  barrier_tensors.push_back(at::empty({1},
    at::TensorOptions().device(at::DeviceType::MLU).dtype(at::kFloat)));
  // All reduce to achieve the barrier
  auto work = allreduce(barrier_tensors);

  // Work will take over barrierTensors
  auto cncl_work = dynamic_cast<ProcessGroupCNCL::WorkCNCL*>(work.get());
  TORCH_CHECK(cncl_work);
  cncl_work->barrier_tensors_ = std::move(barrier_tensors);

  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */, const GatherOptions& /* unused */) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::send(
    std::vector<at::Tensor>& tensors, int dst_rank, int /* unused */) {
  check_mlu_tensors(tensors);
  auto devices = getDeviceList(tensors);
  auto key = getKeySendRecv(rank_, dst_rank);
  int p2p_rank = rank_ <= dst_rank ? 0 : 1;
  int cncl_dst_rank = rank_ <= dst_rank ? 1 : 0;
  bool is_send_recv_self = rank_ == dst_rank;
  if (is_send_recv_self) {
    throw std::runtime_error("Cncl does not support p2p commucation on one deivce");
  }
  auto& cncl_comms = getCNCLComms(key, devices, true, p2p_rank);
  syncQueues(devices[0], cncl_notifier_, cncl_queue_);

  // Work itself will create the CNCL events on all MLUs of tensors
  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  auto input_impl = torch_mlu::getMluTensorImpl(tensors[0]);
  auto input_ptr = input_impl->mlu_data_ptr();

  torch_mlu::recordQueue(tensors[0].storage().data_ptr(), cncl_queue_);
  C10D_CNCL_CHECK(cnclSend(input_ptr, tensors[0].numel(),
                           getCnclDataType(tensors[0].scalar_type()), cncl_dst_rank,
                           cncl_comms->getCnclComm(0), cncl_queue_.queue()));
  work->notifier_.place(cncl_queue_);
  work->blockingWait_ = blockingWait_;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::recv(
    std::vector<at::Tensor>& tensors, int src_rank, int /* unused */) {
  check_mlu_tensors(tensors);
  auto devices = getDeviceList(tensors);
  auto key = getKeySendRecv(rank_, src_rank);
  int p2p_rank = rank_ <= src_rank ? 0 : 1;
  int cncl_src_rank = rank_ <= src_rank ? 1 : 0;
  bool is_send_recv_self = rank_ == src_rank;
  if (is_send_recv_self) {
    throw std::runtime_error("cncl does not support p2p commucation on one deivce");
  }
  auto& cncl_comms = getCNCLComms(key, devices, true, p2p_rank);
  syncQueues(devices[0], cncl_notifier_, cncl_queue_);

  // Work itself will create the CNCL events on all MLUs of tensors
  auto work = c10::make_intrusive<ProcessGroupCNCL::WorkCNCL>(devices);

  auto input_impl = torch_mlu::getMluTensorImpl(tensors[0]);
  auto input_ptr = input_impl->mlu_data_ptr();

  torch_mlu::recordQueue(tensors[0].storage().data_ptr(), cncl_queue_);
  C10D_CNCL_CHECK(cnclRecv(input_ptr, tensors[0].numel(),
                           getCnclDataType(tensors[0].scalar_type()), cncl_src_rank,
                           cncl_comms->getCnclComm(0), cncl_queue_.queue()));
  work->notifier_.place(cncl_queue_);
  work->blockingWait_ = blockingWait_;
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCNCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */, int /* unused */) {
  throw std::runtime_error("Not supported yet");
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupCNCL::createProcessGroupCNCL(
    const c10::intrusive_ptr<::c10d::Store> &store,
    int rank,
    int size,
    const std::chrono::duration<float> &timeout) {
  return c10::make_intrusive<ProcessGroupCNCL>(store, rank, size);
}

}  // namespace c10d
