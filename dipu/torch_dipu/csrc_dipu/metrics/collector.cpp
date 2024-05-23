#include "collector.h"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace dipu::metrics {

void metrics_source::before_update() {}
void metrics_source::after_update() {}

auto metrics_collector::insert_source(metrics_source& s) -> void {
  std::scoped_lock _(mutex);
  sources.insert(&s);
}

auto metrics_collector::remove_source(metrics_source& s) noexcept -> void {
  try {
    std::scoped_lock _(mutex);
    sources.erase(&s);
  } catch (std::exception& ex) {
    std::cerr << "exception during metrics_collector::remove_source: "
              << ex.what() << "\n";
  }
}

auto metrics_collector::fetch() -> statistics {
  std::scoped_lock _(mutex);
  auto target = statistics{};

  for (auto source : sources) {
    source->before_update();
  }

  for (auto source : sources) {
    source->update(target);
  }

  for (auto source : sources) {
    source->after_update();
  }

  return target;
}

}  // namespace dipu::metrics
