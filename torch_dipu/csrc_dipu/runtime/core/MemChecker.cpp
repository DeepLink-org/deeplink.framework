// Copyright (c) 2023, DeepLink.
#include "MemChecker.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <c10/util/Exception.h>
#include <c10/util/Backtrace.h>

namespace dipu {

static const int32_t DEFAULT_MAX_BLOCK_NUM = 10000;
static const int32_t DEFAULT_LOG_INTERVAL = 1000;

MemChecker::~MemChecker() {
  if (!enable()) {
    return;
  }

  if (!blocks_.empty()) {
    std::cout << "dipu memory checker: there maybe exist memory leak. "
      << blocks_.size() << " blocks not released." << std::endl;
    for (const auto &kv : blocks_) {
      std::cout << "key: " << kv.first << ", ptr: " << kv.second.first << ", trace: " << kv.second.second << std::endl;
    }
  }
  std::cout << "dipu memory checker: going to destruction. " << current_state() << std::endl;
}

MemChecker& MemChecker::instance() {
  static MemChecker checker;
  return checker;
}

bool MemChecker::enable() {
  static bool enable = (std::getenv("DIPU_MEM_CHECK") != nullptr);
  return enable;
}

bool MemChecker::enable_backtrace() {
  static bool enable_trace = (std::getenv("DIPU_MEM_CHECK_ENABLE_BACKTRACE") != nullptr);
  return enable_trace;
}

int32_t MemChecker::max_block_num() {
  static int32_t max_block = []() -> int32_t {
    const char* str = std::getenv("DIPU_MEM_CHECK_MAX_BLOCK");
    if (str == nullptr) {
      return DEFAULT_MAX_BLOCK_NUM;
    }
    return std::stoi(str);
  }();

  return max_block;
}

int32_t MemChecker::log_interval() {
  static int32_t interval = []() -> int32_t {
    const char* str = std::getenv("DIPU_MEM_CHECK_LOG_INTERVAL");
    if (str == nullptr) {
      return DEFAULT_LOG_INTERVAL;
    }
    return std::stoi(str);
  }();

  return interval;
}

std::string MemChecker::current_state() const {
  std::stringstream stream;
  stream << "current block num = " <<  blocks_.size()
    << ", total_size = " << (total_size_ >> 20) << "MB"
    << ", insert count = " << insert_cnt_
    << ", max block num = " << max_block_num()
    << ", log interval = " << log_interval();
  return stream.str();
}

void MemChecker::insert(const void* ptr, size_t size) {
  if (!enable() || ptr == nullptr) {
    return;
  }

  bool may_leak = false;
  bool print_log = false;
  std::string state;
  {
    std::lock_guard<std::mutex> lck(mtx_);
    blocks_[ptr] = std::make_pair(size, enable_backtrace() ? c10::get_backtrace() : "");
    total_size_ += static_cast<int64_t>(size);
    ++insert_cnt_;

    if (blocks_.size() > max_block_num()) {
      may_leak = true;
    } else if (insert_cnt_ % log_interval() == 0) {
      print_log = true;
    }

    if (may_leak || print_log) {
      state = current_state();
    }
  }

  if (may_leak) {
    std::cout << "dipu memory checker: there may be memory leak. " << state << std::endl;
  } else if (print_log) {
    std::cout << "dipu memory checker: " << state << std::endl;
  }
}

void MemChecker::erase(const void* ptr) {
  if (!enable() || ptr == nullptr) {
    return;
  }

  bool found = true;
  {
    std::lock_guard<std::mutex> lck(mtx_);
    auto iter = blocks_.find(ptr);
    if (iter == blocks_.end()) {
      found = false;
    } else {
      total_size_ -= static_cast<int64_t>(iter->second.first);
      blocks_.erase(iter);
    }
  }

  if (!found) {
    std::cout << "dipu memory checker: not found point address going to free, ptr = " << ptr << std::endl;
  }
}

void MemChecker::check(const at::Tensor &input) {
  const void *ptr = input.unsafeGetTensorImpl()->unsafe_storage().data();
  check(ptr);
}

void MemChecker::check(const void* ptr) {
  if (!enable()) {
    return;
  }

  bool found = true;
  {
    std::lock_guard<std::mutex> lck(mtx_);
    found = (blocks_.find(ptr) != blocks_.end());
  }

  if (!found) {
    std::cout << "dipu memory checker: not found point address when check, ptr = " << ptr << std::endl;
  }
}

}  // namespace dipu