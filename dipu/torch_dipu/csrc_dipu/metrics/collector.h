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
using exported_histogram = std::pair<
    std::variant<std::vector<exported_integer>, std::vector<exported_floating>>,
    std::vector<exported_integer>>;

namespace detail {
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
}  // namespace detail

template <typename I>
using to_exported_type = typename detail::to_exported_type<I>::type;

template <typename I>
auto to_exported(counter<I> const& v) {
  return static_cast<to_exported_type<I>>(v.get());
}

template <typename I>
auto to_exported(gauge<I> const& v) {
  return static_cast<to_exported_type<I>>(v.get());
}

template <typename I>
auto to_exported(histogram<I> const& v) -> exported_histogram {
  using type = to_exported_type<I>;

  auto& thresholds = v.get_thresholds();
  auto output = std::vector<type>();
  output.reserve(thresholds.size());
  for (auto threshold : thresholds) {
    output.emplace_back(static_cast<type>(threshold));
  }
  return {output, v.template get_buckets<exported_integer>()};
}
}  // namespace dipu::metrics

namespace dipu::metrics {

template <typename C>
struct exported_value {
  using number = exported_number;
  using integer = exported_integer;
  using floating = exported_floating;
  using histogram = exported_histogram;

  std::basic_string<C> name;
  std::basic_string<C> type;
  std::vector<std::pair<std::basic_string<C>, std::basic_string<C>>> labels;
  std::variant<integer, floating, histogram> value;
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
  using integer = typename exported_value<C>::integer;
  using floating = typename exported_value<C>::floating;

 private:
  template <typename... T>
  struct build_variant {
    using type = std::variant<std::shared_ptr<group<T, C>>...>;
  };

  using variant = build_variant<        //
      counter<integer>,                 //
      gauge<integer>, gauge<floating>,  //
      histogram<integer>, histogram<floating>>;
  using variant_type = typename variant::type;

  std::shared_mutex mutable mutex;
  std::unordered_map<std::basic_string<C>, variant_type> groups;

 public:
  [[nodiscard]] auto export_values() const -> std::vector<exported_value<C>> {
    auto output = std::vector<exported_value<C>>{};
    auto visitor = [&output](auto&& group) {
      group->each([&output, &group](labelset<C> const& labels, auto& value) {
        output.emplace_back(exported_value<C>{
            group->name(), group->type(), labels.labels(), to_exported(value)});
      });
    };

    std::shared_lock _(mutex);
    for (auto& [name, group] : groups) {
      std::visit(visitor, group);
    }
    return output;
  }

  template <typename... T>
  [[nodiscard]] auto make_integer_counter(std::basic_string<C> name,
                                          std::basic_string<C> help,
                                          T&&... args)
      -> labeled_value<counter<integer>, C>  //
  {
    return make_type<counter<integer>>("counter<integer> (index: 0)",
                                       std::move(name), std::move(help),
                                       std::forward<T>(args)...);
  }

  template <typename... T>
  [[nodiscard]] auto make_integer_gauge(std::basic_string<C> name,
                                        std::basic_string<C> help, T&&... args)
      -> labeled_value<gauge<integer>, C>  //
  {
    return make_type<gauge<integer>>("gauge<integer> (index: 1)",
                                     std::move(name), std::move(help),
                                     std::forward<T>(args)...);
  }

  template <typename... T>
  [[nodiscard]] auto make_floating_gauge(std::basic_string<C> name,
                                         std::basic_string<C> help, T&&... args)
      -> labeled_value<gauge<floating>, C>  //
  {
    return make_type<gauge<floating>>("gauge<floating> (index: 2)",
                                      std::move(name), std::move(help),
                                      std::forward<T>(args)...);
  }

  [[nodiscard]] auto make_integer_histogram(
      std::basic_string<C> name, std::basic_string<C> help,
      std::initializer_list<exported_integer> list)
      -> labeled_value<histogram<integer>, C>  //
  {
    return make_type<histogram<integer>>("histogram<integer> (index: 3)",
                                         std::move(name), std::move(help),
                                         list);
  }

  template <typename... T>
  [[nodiscard]] auto make_integer_histogram(std::basic_string<C> name,
                                            std::basic_string<C> help,
                                            T&&... args)
      -> labeled_value<histogram<integer>, C>  //
  {
    return make_type<histogram<integer>>("histogram<integer> (index: 3)",
                                         std::move(name), std::move(help),
                                         std::forward<T>(args)...);
  }

  [[nodiscard]] auto make_floating_histogram(
      std::basic_string<C> name, std::basic_string<C> help,
      std::initializer_list<floating> list)
      -> labeled_value<histogram<floating>, C>  //
  {
    return make_type<histogram<floating>>("histogram<floating> (index: 4)",
                                          std::move(name), std::move(help),
                                          list);
  }

  template <typename... T>
  [[nodiscard]] auto make_floating_histogram(std::basic_string<C> name,
                                             std::basic_string<C> help,
                                             T&&... args)
      -> labeled_value<histogram<floating>, C>  //
  {
    return make_type<histogram<floating>>("histogram<floating> (index: 4)",
                                          std::move(name), std::move(help),
                                          std::forward<T>(args)...);
  }

 private:
  template <typename T, typename... U>
  [[nodiscard]] auto make_type(            //
      char const* type,                    //
      std::basic_string<C> name,           //
      U&&... args) -> labeled_value<T, C>  //
  {
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
      throw std::runtime_error("unexpected nullptr shared pointer");
    }

    return (**pointer).find_or_make({});
  }

  template <typename T, typename... U>
  auto make_variant(std::basic_string<C> name, U&&... args) -> variant_type& {
    {
      std::shared_lock _(mutex);
      if (auto iter = groups.find(name); iter != groups.end()) {
        return iter->second;
      }
    }
    {
      std::unique_lock _(mutex);
      using type = group<T, C>;
      auto dummy = std::shared_ptr<type>();
      auto [iter, done] = groups.try_emplace(name, dummy);
      if (done) {
        iter->second = std::make_shared<type>(name, std::forward<U>(args)...);
      }
      return iter->second;
    }
  }
};

}  // namespace dipu::metrics
