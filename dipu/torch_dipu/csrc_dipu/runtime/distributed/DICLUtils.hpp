// Copyright (c) 2023, DeepLink.
#pragma once

#include <algorithm>

#include <c10/core/Device.h>

#include <csrc_dipu/runtime/devproxy/deviceproxy.h>
#include <csrc_dipu/runtime/devproxy/diclproxy.h>

namespace dipu {

// wrapper of vendor raw communicator
class DICLComm {
 private:
  void initRawComm(int numRanks, int rank, commUniqueId uniqueid) {
    devproxy::diclCommInitRank(&rawComm_, numRanks, uniqueid, rank,
                               static_cast<int>(device_.index()));
  }

 public:
  explicit DICLComm(DIPUStream &bindStream)
      : diclStream_(bindStream), device_(bindStream.device()) {}

  ~DICLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    std::unique_lock<std::mutex> lock(mutex_);
    if (rawComm_ && !aborted_) {
      devproxy::diclCommDestroy(rawComm_);
      rawComm_ = nullptr;
    }
  }
  static std::shared_ptr<DICLComm> create(int numRanks, int rank,
                                          commUniqueId uniqueid,
                                          DIPUStream &stream) {
    auto comm = std::make_shared<DICLComm>(stream);
    comm->initRawComm(numRanks, rank, uniqueid);
    return comm;
  }

  // Must not be copyable
  DICLComm(const DICLComm &) = delete;
  DICLComm &operator=(const DICLComm &) = delete;

  // Move constructable
  DICLComm(DICLComm &&other) = delete;
  // Move assignable
  DICLComm &operator=(DICLComm &&other) = delete;

  diclComm_t rawComm() const { return rawComm_; }

  void preSyncStream() {
    auto currStream = dipu::getCurrentDIPUStream(device_.index());
    preEvent_.record(currStream);
    preEvent_.wait(diclStream_);
  }

  // The DIPU queues used by DICL kernels
  DIPUStream diclStream_;
  // The DIPU events used to sync current stream
  DIPUEvent preEvent_;

  // by default, copy should work in comm stream, if in other stream, use
  // preCopyEvent_ to guarantee comm finish.
  DIPUEvent preCopyEvent_;

  // The cached list of DIPU devices to operate on
  at::Device device_;

 protected:
  bool aborted_ = false;
  diclComm_t rawComm_ = nullptr;
  mutable std::mutex mutex_;
};

}  // namespace dipu