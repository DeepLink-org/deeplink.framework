#include "default.h"

#include <algorithm>
#include <random>
#include <string>
#include <string_view>

#include "collector.h"
#include "source.h"

namespace dipu {

auto generate_random_alphanum(std::size_t length) -> std::string {
  auto static thread_local engine = std::mt19937{std::random_device()()};
  auto static constexpr charset = std::string_view(
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz");

  auto out = std::string(length, charset.front());
  std::generate(out.begin(), out.end(), []() {
    using uniform = std::uniform_int_distribution<std::size_t>;
    auto index = uniform(0, charset.size())(engine);
    return charset[index];
  });
  return out;
}

auto default_metrics_collector() -> metrics::metrics_collector& {
  auto static instance = metrics::metrics_collector();
  return instance;
}

auto default_device_allocator_metrics_producer()
    -> metrics::allocator_metrics_producer<true>& {
  auto static instance = metrics::allocator_metrics_producer<true>(
      "allocator.device", default_metrics_collector());
  return instance;
}

auto default_host_allocator_metrics_producer()
    -> metrics::allocator_metrics_producer<true>& {
  auto static instance = metrics::allocator_metrics_producer<true>(
      "allocator.host", default_metrics_collector());
  return instance;
}

}  // namespace dipu
