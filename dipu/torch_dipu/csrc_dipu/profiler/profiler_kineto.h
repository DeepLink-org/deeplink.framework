#pragma once

#include <memory>
#include <string>
#include <set>
#include <unordered_set>

#include <ATen/record_function.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/autograd/profiler_kineto.h>

namespace dipu {
namespace profile {

void prepareProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities);

void enableProfiler(
    const torch::profiler::impl::ProfilerConfig& config,
    const std::set<torch::profiler::impl::ActivityType>& activities,
    const std::unordered_set<at::RecordScope>& scopes = {});

std::unique_ptr<torch::autograd::profiler::ProfilerResult> disableProfiler();

void addMetadataJson(const std::string& key, const std::string& value);

void profilerStep();

}  // namespace profile
}  // namespace dipu