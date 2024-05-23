#pragma once

#include <cstdint>
#include <unordered_map>
#include <variant>

#include "numeric.h"

namespace dipu::metrics {

struct allocator_statistics;
using statistics_value = std::variant<allocator_statistics>;
using statistics = std::unordered_map<std::string, statistics_value>;

struct allocator_statistics {
  // TODO(wy): collect duration
  //
  // using time_type = std::chrono::microseconds;
  // enum { default_samples = 1'000 };

  maximum_minimum_summation<uint64_t> allocate_size;
  fixed_size_exponential_histogram<uint64_t, 2> allocate_size_frequency;
  std::size_t allocate_nullptr_count{};
  std::size_t allocate_duplicated_count{};

  maximum_minimum_summation<uint64_t> deallocate_size;
  fixed_size_exponential_histogram<uint64_t, 2> deallocate_size_frequency;
  std::size_t deallocate_nullptr_count{};
  std::size_t deallocate_unexpected_count{};

  uint64_t used_size_summation{};
  fixed_size_exponential_histogram<uint64_t, 2> used_size_frequency;
};

}  // namespace dipu::metrics
