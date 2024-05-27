#include "export.h"

#include <memory>

#include "collector.h"
#include "label.h"

namespace dipu {

auto default_metrics_collector() -> metrics::collector<char>& {
  auto static instance = new metrics::collector<char>();
  return *instance;
}

auto default_device_allocator_metrics_producer() -> allocator_metrics& {
  auto static instance = new allocator_metrics(
      default_metrics_collector(), metrics::labelset<char>{{"type", "device"}});
  return *instance;
}

auto default_host_allocator_metrics_producer() -> allocator_metrics& {
  auto static instance = new allocator_metrics(
      default_metrics_collector(), metrics::labelset<char>{{"type", "host"}});
  return *instance;
}

}  // namespace dipu
