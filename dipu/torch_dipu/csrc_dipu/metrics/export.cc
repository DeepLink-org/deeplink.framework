#include "export.h"

// "extenal" is necessary in order to make sure pybind11 finds the same
// collector instance.
auto dipu::metrics::default_collector() -> Collector& {
  auto static instance = metrics::Collector();
  return instance;
}
