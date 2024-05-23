#pragma once

#include <mutex>
#include <unordered_set>

#include "statistics.h"

namespace dipu::metrics {

class metrics_source {
 public:
  virtual void before_update();
  virtual void after_update();
  virtual void update(statistics& target) = 0;

  // :-(
  virtual ~metrics_source() = default;
  metrics_source() = default;
  metrics_source(metrics_source&&) = default;
  metrics_source(metrics_source const&) = default;
  metrics_source& operator=(metrics_source&&) = default;
  metrics_source& operator=(metrics_source const&) = default;
};

class metrics_collector {
  std::mutex mutable mutex;
  std::unordered_set<metrics_source*> sources;

 public:
  auto insert_source(metrics_source& s) -> void;
  auto remove_source(metrics_source& s) noexcept -> void;
  auto fetch() -> statistics;
};

}  // namespace dipu::metrics
