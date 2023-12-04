// Copyright (c) 2023, DeepLink.
#pragma once

#include <mutex>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>

#include <ATen/Tensor.h>

namespace dipu {

class MemChecker final {
 public:
  static MemChecker& instance();
  static bool enable();
  static bool enable_backtrace();
  static int32_t max_block_num();
  static int32_t log_interval();

  void insert(const void* ptr, size_t size);
  void erase(const void* ptr);
  void check(const at::Tensor& input);
  void check(const void* ptr);
  ~MemChecker();

 private:
  std::string current_state() const;

  std::mutex mtx_;
  std::unordered_map<const void*, std::pair<size_t, std::string>> blocks_;
  int64_t total_size_ = 0;
  int64_t insert_cnt_ = 0;
};

}  // namespace dipu
