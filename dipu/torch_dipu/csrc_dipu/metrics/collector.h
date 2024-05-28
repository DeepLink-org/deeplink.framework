#pragma once

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "group.h"
#include "value.h"

namespace dipu::metrics {

using exported_floating = double;
using exported_integer = int64_t;
using exported_number = std::variant<exported_integer, exported_floating>;
using exported_histogram = std::pair<  // A pair of thresholds and buckets
    std::variant<std::vector<exported_integer>,
                 std::vector<exported_floating>>,  // Thresholds (ints or reals)
    std::vector<exported_integer>                  // Buckets
    >;

namespace detail {

// A trait to mapping type I to exported type
template <typename I, typename = void>
struct to_exported_type {
  // value type not supported
};

template <typename I>
struct to_exported_type<I, std::enable_if_t<std::is_floating_point_v<I>>> {
  using type = exported_floating;
};

template <typename I>
struct to_exported_type<I, std::enable_if_t<std::is_integral_v<I>>> {
  using type = exported_integer;
};

template <typename I>
using to_exported_t = typename to_exported_type<I>::type;

// Convert counter to exported value
template <typename I>
auto to_exported(counter<I> const& v) -> to_exported_t<I> {
  return static_cast<to_exported_t<I>>(v.get());
}

// Convert gauge to exported value
template <typename I>
auto to_exported(gauge<I> const& v) -> to_exported_t<I> {
  return static_cast<to_exported_t<I>>(v.get());
}

// Convert histogram to exported histogram
template <typename I>
auto to_exported(histogram<I> const& v) -> exported_histogram {
  using type = to_exported_t<I>;
  auto& thresholds = v.get_thresholds();
  auto output = std::vector<type>();
  output.reserve(thresholds.size());
  for (auto threshold : thresholds) {
    output.emplace_back(static_cast<type>(threshold));
  }
  return {output, v.template get_buckets<exported_integer>()};
}

}  // namespace detail
}  // namespace dipu::metrics

namespace dipu::metrics {

template <typename C>
struct exported_group {
  using number = exported_number;
  using integer = exported_integer;
  using floating = exported_floating;
  using histogram = exported_histogram;
  using labels = std::vector<label<C>>;
  using value = std::variant<integer, floating, histogram>;
  using labeled_value = std::pair<labels, value>;

  std::basic_string<C> name;
  std::basic_string<C> type;
  std::basic_string<C> info;
  std::vector<labeled_value> values;
};

using labeled_integer_counter = labeled_value<counter<exported_integer>, char>;
using labeled_integer_gauge = labeled_value<gauge<exported_integer>, char>;
using labeled_floating_gauge = labeled_value<gauge<exported_floating>, char>;
using labeled_integer_histogram =
    labeled_value<histogram<exported_integer>, char>;
using labeled_floating_histogram =
    labeled_value<histogram<exported_floating>, char>;

template <typename C>
class collector {
 public:
  template <typename T>
  using labeled = labeled_value<T, C>;
  using exported = exported_group<C>;
  using floating = typename exported::floating;
  using integer = typename exported::integer;
  using string = std::basic_string<C>;

 private:
  // Map {T...} to {std::shared_ptr< group<T> >...}
  template <typename... T>
  struct to_shared_grouped_variant {
    using type = std::variant<std::shared_ptr<group<T, C>>...>;
  };

  using variant_type = typename to_shared_grouped_variant<
      counter<integer>,                        // 0: counter
      gauge<integer>, gauge<floating>,         // 1, 2: gauge
      histogram<integer>, histogram<floating>  // 3, 4: histogram
      >::type;

  std::shared_mutex mutable mutex;
  std::unordered_map<string, variant_type> groups;

 public:
  [[nodiscard]] auto export_values() const -> std::vector<exported> {
    auto output = std::vector<exported>{};
    auto visitor = [&output](auto&& group) {
      auto values = std::vector<typename exported::labeled_value>();
      group->each([&values](labelset<C> const& key, auto& value) {
        values.emplace_back(key.labels(), detail::to_exported(value));
      });
      output.push_back({group->name(), group->type(), group->description(),
                        std::move(values)});
    };

    // Iterate through groups and collect values in each group.
    std::shared_lock _(mutex);
    output.reserve(groups.size());
    for (auto& [name, group] : groups) {
      std::visit(visitor, group);
    }
    return output;
  }

  template <typename... T>
  [[nodiscard]] auto make_integer_counter(string name, string help, T&&... args)
      -> labeled<counter<integer>>  //
  {
    return make_type<counter<integer>>("counter<integer> (index: 0)",
                                       std::move(name), std::move(help),
                                       std::forward<T>(args)...);
  }

  template <typename... T>
  [[nodiscard]] auto make_integer_gauge(string name, string help, T&&... args)
      -> labeled<gauge<integer>>  //
  {
    return make_type<gauge<integer>>("gauge<integer> (index: 1)",
                                     std::move(name), std::move(help),
                                     std::forward<T>(args)...);
  }

  template <typename... T>
  [[nodiscard]] auto make_floating_gauge(string name, string help, T&&... args)
      -> labeled<gauge<floating>>  //
  {
    return make_type<gauge<floating>>("gauge<floating> (index: 2)",
                                      std::move(name), std::move(help),
                                      std::forward<T>(args)...);
  }

  [[nodiscard]] auto make_integer_histogram(
      string name, string help,
      std::initializer_list<exported_integer> list)
      -> labeled<histogram<integer>>  //
  {
    return make_type<histogram<integer>>("histogram<integer> (index: 3)",
                                         std::move(name), std::move(help),
                                         list);
  }

  template <typename... T>
  [[nodiscard]] auto make_integer_histogram(string name, string help,
                                            T&&... args)
      -> labeled<histogram<integer>>  //
  {
    return make_type<histogram<integer>>("histogram<integer> (index: 3)",
                                         std::move(name), std::move(help),
                                         std::forward<T>(args)...);
  }

  [[nodiscard]] auto make_floating_histogram(
      string name, string help,
      std::initializer_list<floating> list) -> labeled<histogram<floating>>  //
  {
    return make_type<histogram<floating>>("histogram<floating> (index: 4)",
                                          std::move(name), std::move(help),
                                          list);
  }

  template <typename... T>
  [[nodiscard]] auto make_floating_histogram(string name, string help,
                                             T&&... args)
      -> labeled<histogram<floating>>  //
  {
    return make_type<histogram<floating>>("histogram<floating> (index: 4)",
                                          std::move(name), std::move(help),
                                          std::forward<T>(args)...);
  }

 private:
  template <typename T, typename... U>
  [[nodiscard]] auto make_type(char const* type, string name, U&&... args)
      -> labeled<T> {
    if (name.empty()) {
      auto m = std::string(type) + " name cannot be empty";
      throw std::invalid_argument(m);
    }

    auto& variant = make_variant<T>(std::move(name), std::forward<U>(args)...);
    auto* pointer = std::get_if<std::shared_ptr<group<T, C>>>(&variant);

    if (pointer == nullptr) {
      auto m = std::string("expect type ") + type + " but index is " +
               std::to_string(variant.index());
      throw std::invalid_argument(m);
    }

    if (*pointer == nullptr) {
      throw std::runtime_error("unexpected null shared pointer");
    }

    return (**pointer).find_or_make({});
  }

  template <typename T, typename... U>
  auto make_variant(string name, U&&... args) -> variant_type& {
    // Return the variant if found, or insert a new variant into gourps.
    {
      std::shared_lock _(mutex);
      if (auto iter = groups.find(name); iter != groups.end()) {
        return iter->second;
      }
    }
    {
      // Avoid invoking std::make_shared if name already existed. Here we use a
      // null pointer as dummy.
      using type = group<T, C>;
      auto dummy = std::shared_ptr<type>();

      std::unique_lock _(mutex);
      auto [iter, inserted] = groups.try_emplace(name, dummy);
      if (inserted) {
        iter->second = std::make_shared<type>(name, std::forward<U>(args)...);
      }
      return iter->second;
    }
  }
};

}  // namespace dipu::metrics
