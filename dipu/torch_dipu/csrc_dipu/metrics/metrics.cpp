#include "metrics.h"

#include <atomic>

auto extern dipu::metrics::enable(std::optional<bool> update) -> bool {
  auto static value = std::atomic_bool(true);
  if (update) {
    value.store(update.value(), std::memory_order_release);
  }
  return value.load(std::memory_order_acquire);
}

// "extenal" is necessary in order to make sure pybind11 finds the same
// collector instance.
auto dipu::metrics::default_collector() -> Collector& {
  auto static instance = metrics::Collector();
  return instance;
}
