// Copyright (c) 2023, DeepLink.
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/library.h>

#include <csrc_dipu/base/basedef.h>

namespace c10d {
namespace ops {

// Below are ProcessGroup's corresponding ops for each backend. Ops are but
// routed through the dispatcher to be dispatched to the appropriate backend.
// Currently a no-op as the process group does not have a list of backends.

c10::intrusive_ptr<Work> send_dipu(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
      ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_dipu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
      ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> reduce_dipu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op, int64_t root_rank,
    int64_t root_tensor, int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
      ->reduce(tensor_vec,
               ReduceOptions{*reduce_op.get(), root_rank, root_tensor,
                             std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_dipu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank,
    int64_t root_tensor, int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
          ->broadcast(tensor_vec,
                      BroadcastOptions{root_rank, root_tensor,
                                       std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_dipu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
#ifdef DIPU_TORCH200
#else
    const c10::optional<at::Tensor>& sparse_indices,
#endif
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
          ->allreduce(tensor_vec,
                      AllreduceOptions{*reduce_op.get(),
                                       std::chrono::milliseconds(timeout)});

  // Return input tensors as output tensors to make inplace allreduce look like
  // a functional API, so that make_fx can correctly build the dependencies in
  // the graph later.
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_dipu_(const std::vector<std::vector<at::Tensor>>& output_tensors,
                at::TensorList input_tensors,
                const c10::intrusive_ptr<ProcessGroup>& process_group,
                int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout)});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::tuple<std::vector<std::vector<at::Tensor>>,
                    c10::intrusive_ptr<Work>>(output_tensors, work);
}

// refer to distributed/c10d/Ops.cpp
std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_dipu_(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto work = process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
                  ->_allgather_base(output_tensor, input_tensor);

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_dipu_(const at::TensorList& output_tensors,
                     const std::vector<std::vector<at::Tensor>>& input_tensors,
                     const c10::intrusive_ptr<ProcessGroup>& process_group,
                     const c10::intrusive_ptr<ReduceOp>& reduce_op,
                     int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{*reduce_op.get(),
                                   std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors_vec, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_dipu_(
    at::Tensor& output_tensor, at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op, int64_t timeout) {
  auto work = process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
                  ->_reduce_scatter_base(
                      output_tensor, input_tensor,
                      ReduceScatterOptions{*reduce_op.get(),
                                           std::chrono::milliseconds(timeout)});

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

c10::intrusive_ptr<Work> gather_dipu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_dipu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group, int64_t root_rank,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

c10::intrusive_ptr<Work> barrier_dipu(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids, int64_t timeout) {
  return process_group->getBackend(dipu::DIPU_DEVICE_TYPE)
      ->barrier(BarrierOptions{device_ids, std::chrono::milliseconds(timeout)});
}

// register functions to dispatcher
TORCH_LIBRARY_IMPL(c10d, DIPU_DEVICE_TYPE_MACRO, m) {
  m.impl("send", send_dipu);
  m.impl("recv_", recv_dipu_);
  m.impl("broadcast_", broadcast_dipu_);
  m.impl("reduce_", reduce_dipu_);
  m.impl("allreduce_", allreduce_dipu_);
  m.impl("allgather_", allgather_dipu_);
  m.impl("_allgather_base_", _allgather_base_dipu_);
  m.impl("scatter_", scatter_dipu_);
  m.impl("reduce_scatter_", reduce_scatter_dipu_);
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_dipu_);
  m.impl("barrier", barrier_dipu);

  // not implement
  m.impl("gather_", gather_dipu_);

  // unregistered op, we expect it can fallback to cpu, but it not work now
  // (hard to sync).
}

}  // namespace ops
}  // namespace c10d
