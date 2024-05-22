// Copyright (c) 2024, DeepLink.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <c10/core/DispatchKey.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "csrc_dipu/base/environ.hpp"

namespace at {

class DipuOpRegister {
 public:
  using RegisterFuncPtr = void (*)(torch::Library&);
  using RegisterClosure = std::function<void()>;

  // use singleton to ensure correct initialization order
  static DipuOpRegister& instance() {
    static DipuOpRegister obj;
    return obj;
  }

  DipuOpRegister(const DipuOpRegister&) = delete;
  DipuOpRegister& operator=(const DipuOpRegister&) = delete;

  void registerOpMaybeDelayed(RegisterFuncPtr fun_ptr, const char* ns,
                              c10::optional<c10::DispatchKey> key,
                              const char* file, uint32_t line) {
    std::lock_guard<std::mutex> guard(mutex_);
    libs_.push_back(std::make_unique<torch::Library>(torch::Library::IMPL, ns,
                                                     key, file, line));
    auto register_closure = [fun_ptr, lib = libs_.back().get()]() {
      fun_ptr(*lib);
    };
    if (dipu::environ::immediateRegisterOp()) {
      register_closure();
    } else {
      delayed_registers_.emplace_back(std::move(register_closure));
    }
  }

  void applyDelayedRegister() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& register_closure : delayed_registers_) {
      register_closure();
    }
    delayed_registers_.clear();
    delayed_registers_.shrink_to_fit();
  }

 private:
  DipuOpRegister() = default;

  std::vector<std::unique_ptr<torch::Library>> libs_;
  std::vector<RegisterClosure> delayed_registers_;
  std::mutex mutex_;
};

class DipuOpRegisterHelper {
 public:
  DipuOpRegisterHelper(DipuOpRegister::RegisterFuncPtr fun_ptr, const char* ns,
                       c10::optional<c10::DispatchKey> key, const char* file,
                       uint32_t line) {
    DipuOpRegister::instance().registerOpMaybeDelayed(fun_ptr, ns, key, file,
                                                      line);
  }
};

}  // namespace at
