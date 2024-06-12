#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "detail/collector.h"
#include "labeled.h"

namespace dipu::metrics {

// Global settings.

// Enable or disable or query state of metrics.
//
// - If value is std::nullopt, return current state.
// - Otherwise, state will be changed.
//
// Warning: metrics values may become no longer accurate once "enable(false)" is
// called.
auto extern enable(std::optional<bool> update = std::nullopt) -> bool;

// The default global collector (singleton)
class Collector;
auto extern default_collector() -> Collector&;

// To simplify use cases, only int64_t and double are used as numbers.
using ExportedInteger = int64_t;
using ExportedFloating = double;
using ExportedHistogram = std::tuple<  // A pair of thresholds and buckets
    std::variant<std::vector<ExportedInteger>,
                 std::vector<ExportedFloating>>,  // Thresholds (ints or reals)
    std::vector<ExportedInteger>,                 // Buckets
    std::variant<ExportedInteger, ExportedFloating>  // Summation
    >;

}  // namespace dipu::metrics

namespace dipu::metrics {

// Default string type is std::string.
using String = std::string;

namespace detail {
// Define currently supported string type (for labels) and metrics value type.
using exportable_collector = collector<String,
                                       counter<ExportedInteger>,    //
                                       counter<ExportedFloating>,   //
                                       gauge<ExportedInteger>,      //
                                       gauge<ExportedFloating>,     //
                                       histogram<ExportedInteger>,  //
                                       histogram<ExportedFloating>>;
}  // namespace detail

// Some type alias to increase code readability.

using LabeledIntegerCounter = LabeledValue<String, counter<ExportedInteger>>;
using LabeledFloatingCounter = LabeledValue<String, counter<ExportedFloating>>;
using LabeledIntegerGauge = LabeledValue<String, gauge<ExportedInteger>>;
using LabeledFloatingGauge = LabeledValue<String, gauge<ExportedFloating>>;
using LabeledIntegerHistogram =
    LabeledValue<String, histogram<ExportedInteger>>;
using LabeledFloatingHistogram =
    LabeledValue<String, histogram<ExportedFloating>>;

// Collector is a factory to create each metric values.
class Collector : public detail::exportable_collector {
  using base = detail::exportable_collector;
  using base::list;
  using base::make;

 public:
  using string = typename base::string;
  using integer = ExportedInteger;
  using floating = ExportedFloating;
  using labelset = detail::labelset<string>;

  // Reset all metrics values.
  //
  // Warning, this action may cause metric values no longer accurate. So use at
  // your own risk.
  auto reset() -> void {
    auto constexpr resetor = [](auto& group) { group->reset(); };
    for (auto& reference : list()) {
      std::visit(resetor, reference.get());
    }
  }

  [[nodiscard]] auto make_integer_counter(string name, string help,
                                          integer initial = 0)
      -> LabeledIntegerCounter {
    return make<LabeledValue<string, counter<integer>>, counter<integer>>(
        "counter<integer> (index: 0)", std::move(name), string("counter"),
        std::move(help), initial);
  }

  [[nodiscard]] auto make_floating_counter(string name, string help,
                                           floating initial = 0)
      -> LabeledFloatingCounter {
    return make<LabeledValue<string, counter<floating>>, counter<floating>>(
        "counter<floating> (index: 1)", std::move(name), string("counter"),
        std::move(help), initial);
  }

  [[nodiscard]] auto make_integer_gauge(string name, string help,
                                        integer initial = 0)
      -> LabeledIntegerGauge {
    return make<LabeledValue<string, gauge<integer>>, gauge<integer>>(
        "gauge<integer> (index: 2)", std::move(name), string("gauge"),
        std::move(help), initial);
  }

  [[nodiscard]] auto make_floating_gauge(string name, string help,
                                         floating initial = 0)
      -> LabeledFloatingGauge {
    return make<LabeledValue<string, gauge<floating>>, gauge<floating>>(
        "gauge<floating> (index: 3)", std::move(name), string("gauge"),
        std::move(help), initial);
  }

  [[nodiscard]] auto make_integer_histogram(string name, string help,
                                            std::initializer_list<integer> list)
      -> LabeledIntegerHistogram {
    return make<LabeledValue<string, histogram<integer>>, histogram<integer>>(
        "histogram<integer> (index: 4)", std::move(name), string("histogram"),
        std::move(help), list);
  }

  template <typename... T>
  [[nodiscard]] auto make_integer_histogram(string name, string help,
                                            T&&... args)
      -> LabeledIntegerHistogram {
    return make<LabeledValue<string, histogram<integer>>, histogram<integer>>(
        "histogram<integer> (index: 4)", std::move(name), string("histogram"),
        std::move(help), std::forward<T>(args)...);
  }

  [[nodiscard]] auto make_floating_histogram(
      string name, string help, std::initializer_list<floating> list)
      -> LabeledFloatingHistogram {
    return make<LabeledValue<string, histogram<floating>>, histogram<floating>>(
        "histogram<floating> (index: 5)", std::move(name), string("histogram"),
        std::move(help), list);
  }

  template <typename... T>
  [[nodiscard]] auto make_floating_histogram(string name, string help,
                                             T&&... args)
      -> LabeledFloatingHistogram {
    return make<LabeledValue<string, histogram<floating>>, histogram<floating>>(
        "histogram<floating> (index: 5)", std::move(name), string("histogram"),
        std::move(help), std::forward<T>(args)...);
  }
};

// Some helper functions or classes for ExportedGroup.
namespace detail {

// A trait to mapping number type I to exported number type.
template <typename I, typename = void>
struct exported_number_type {};
template <typename I>
struct exported_number_type<I, std::enable_if_t<std::is_floating_point_v<I>>> {
  using type = ExportedFloating;
};
template <typename I>
struct exported_number_type<I, std::enable_if_t<std::is_integral_v<I>>> {
  using type = ExportedInteger;
};
template <typename I>
using exported_number_t = typename exported_number_type<I>::type;

// Convert counter to exported value
template <typename I>
auto to_exported_number(counter<I> const& v) -> exported_number_t<I> {
  return static_cast<exported_number_t<I>>(v.get());
}

// Convert gauge to exported value
template <typename I>
auto to_exported_number(gauge<I> const& v) -> exported_number_t<I> {
  return static_cast<exported_number_t<I>>(v.get());
}

// Convert histogram to exported histogram
template <typename I>
auto to_exported_number(histogram<I> const& v) -> ExportedHistogram {
  using type = exported_number_t<I>;
  auto& thresholds = v.get_thresholds();
  auto output = std::vector<type>();
  output.reserve(thresholds.size());
  for (auto threshold : thresholds) {
    output.emplace_back(static_cast<type>(threshold));
  }
  return {output, v.template get_buckets<ExportedInteger>(), v.sum()};
}

}  // namespace detail

// ExportedGroup is used to convert metrics value into STL containers. Thus it
// becomes easier for Pybind11 to convert them into Python basic types.
//
// ExportedGroup self is register to Pybind11 directly.
//
// As it follows Prometheus style metrics. Each metrics value contains at least
// five parts (timestamp is not included here):
// 1. name, 2. type, 3. description, 4. labels 5. value
//
// By design, a group of metrics values have same name, type and description.
// So they are grouped together into ExportedGroup.
struct ExportedGroup {
  using string = std::string;
  using integer = ExportedInteger;
  using floating = ExportedFloating;
  using histogram = ExportedHistogram;
  using labels = std::vector<std::pair<string, string>>;
  using value = std::variant<integer, floating, histogram>;
  using labeled_value = std::pair<labels, value>;

  string name;
  string type;
  string info;
  std::vector<labeled_value> values;

  [[nodiscard]] auto static from_collector(
      detail::exportable_collector const& collector)
      -> std::vector<ExportedGroup> {
    auto output = std::vector<ExportedGroup>{};
    output.reserve(collector.size());
    auto visitor = [&output](auto const& group) {
      auto values = std::vector<labeled_value>();
      values.reserve(group->size());
      auto filler = [&values](auto const& key, auto const& value) {
        values.emplace_back(key.labels(), detail::to_exported_number(value));
      };
      group->for_each(filler);
      output.push_back({group->name(),         //
                        group->type(),         //
                        group->description(),  //
                        std::move(values)});
    };

    for (auto& reference : collector.list()) {
      std::visit(visitor, reference.get());
    }
    return output;
  }
};

}  // namespace dipu::metrics
