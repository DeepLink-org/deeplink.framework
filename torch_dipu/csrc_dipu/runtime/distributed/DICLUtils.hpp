
#pragma once

#include <algorithm>

#include <csrc_dipu/runtime/device/diclapis.h>

namespace dipu {

// wrapper of vendor raw communicator
class DICLComm {
public:
  explicit DICLComm(diclComm_t rawComm) : rawComm_(rawComm) {}

  DICLComm() : DICLComm(nullptr) {}

  ~DICLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    std::unique_lock<std::mutex> lock(mutex_);
    if (rawComm_ && !aborted_) {
      devapis::diclCommDestroy(rawComm_);
      rawComm_ = nullptr;
    }
  }
  static std::shared_ptr<DICLComm> create(int numRanks, int rank, commUniqueId_t uniqueid, int localDeviceId = -1) {
    auto comm = std::make_shared<DICLComm>();
    devapis::diclCommInitRank(&(comm->rawComm_), numRanks, uniqueid, rank, localDeviceId);
    return comm;
  }

  // Must not be copyable
  DICLComm(const DICLComm&) = delete;
  DICLComm& operator=(const DICLComm&) = delete;

  // Move constructable
  DICLComm(DICLComm&& other) {
    std::swap(rawComm_, other.rawComm_);
  }
  // Move assignable
  DICLComm& operator=(DICLComm&& other) {
    std::swap(rawComm_, other.rawComm_);
    return *this;
  }

  diclComm_t rawComm() const{
    return rawComm_;
  }

protected:
  bool aborted_ = false;
  diclComm_t rawComm_ = nullptr;
  mutable std::mutex mutex_;
};

} // namespace dipu