#pragma once

#include <string>

#include "collector.h"
#include "source.h"

namespace dipu {

auto extern generate_random_alphanum(std::size_t length) -> std::string;
auto extern default_metrics_collector() -> metrics::metrics_collector&;
auto extern default_device_allocator_metrics_producer()
    -> metrics::allocator_metrics_producer<true>&;
auto extern default_host_allocator_metrics_producer()
    -> metrics::allocator_metrics_producer<true>&;

}  // namespace dipu
