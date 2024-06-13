#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include "label.h"

namespace dipu::metrics::detail {

// shared_value is internal wrapper for other value type (Counter, Gauge...)
// inside Group.
//
// Warning: shared_value is managed by Group, should not use directly by user.
template <typename T>
struct shared_value {
  T value;
  std::atomic_bool use{};  // TODO(cxx20): upgrade to std::atomic_flag
  std::atomic_uint count{};

  explicit shared_value(T const& source) noexcept : value(source) {}
  auto incref() noexcept -> void { count.fetch_add(1); }
  auto decref() noexcept -> void { count.fetch_sub(1); }
  auto touch() noexcept -> T& {
    use.store(true, std::memory_order_release);
    return value;
  }
  auto unused() const noexcept -> bool {
    return not use.load(std::memory_order_acquire) and count.load() == 0;
  }
};

// group is managed by collector, and should not use directly by user.
template <typename S /* string type */, typename T /* value type */>
class group : public std::enable_shared_from_this<group<S, T>> {
  S group_name;
  S group_type;
  S group_description;
  T default_value;
  std::shared_mutex mutable mutex;
  std::unordered_map<labelset<S>, shared_value<T>> values;

 public:
  using value_type = typename decltype(values)::value_type;

  template <typename... U>
  explicit group(S name, S type, S help, U&&... args) noexcept
      : group_name(std::move(name)),         //
        group_type(std::move(type)),         //
        group_description(std::move(help)),  //
        default_value(std::forward<U>(args)...) {}

  auto name() const noexcept -> S const& { return group_name; }
  auto type() const noexcept -> S const& { return group_type; }
  auto description() const noexcept -> S const& { return group_description; }

  template <typename U /* LabeledValue */>
  [[nodiscard]] auto make(labelset<S> labels) -> U {
    {  // return created if found
      std::shared_lock _(mutex);
      if (auto iter = values.find(labels); iter != values.end()) {
        return U(this->shared_from_this(), *iter);
      }
    }
    {  // return a newly created shared_value if not found
      std::unique_lock _(mutex);
      auto [iter, done] = values.try_emplace(std::move(labels), default_value);
      return U(this->shared_from_this(), *iter);
    }
  }

  template <typename F>
  auto for_each(F f) -> void {
    static_assert(std::is_invocable_v<F, labelset<S> const&, T&> ||
                  std::is_invocable_v<F, labelset<S> const&, T const&>);

    auto found_unused = false;
    {
      std::shared_lock _(mutex);
      for (auto& [key, value] : values) {
        if (value.unused()) {
          found_unused = true;
        } else {
          f(key, value.value);
        }
      }
    }

    if (found_unused) {
      std::unique_lock _(mutex);
      for (auto iter = values.begin(); iter != values.end();) {
        if (iter->second.unused()) {
          iter = values.erase(iter);
        } else {
          ++iter;
        }
      }
    }
  }

  [[nodiscard]] auto size() const -> std::size_t {
    std::shared_lock _(mutex);
    return values.size();
  }
};

// collector should not be used directly. Please see Collector (outside
// detail namespace).
template <typename S, typename... V>
class collector {
 public:
  using string = S;
  using variant = std::variant<std::shared_ptr<group<S, V>>...>;

 private:
  std::shared_mutex mutable mutex;
  std::unordered_map<S, variant> groups;

 public:
  // In general, U should be LabeledValue.
  template <typename U, typename T, typename... A>
  [[nodiscard]] auto make(char const* hint, S name, A&&... args) -> U {
    static_assert((std::is_same_v<T, V> || ...), "V must be one of {T, ...}");

    if (name.empty()) {
      auto m = std::string(hint) + " name cannot be empty";
      throw std::invalid_argument(m);
    }

    // Find or create a named variant group.
    auto& named = make_variant<T>(std::move(name), std::forward<A>(args)...);
    auto* which = std::get_if<std::shared_ptr<group<S, T>>>(&named);

    if (which == nullptr) {
      auto m = std::string("expect type ") + hint + " but index is " +
               std::to_string(named.index());
      throw std::invalid_argument(m);
    }

    if (*which == nullptr) {
      throw std::runtime_error("unexpected null shared pointer");
    }

    return (**which).template make<U>({});
  }

  [[nodiscard]] auto list() const
      -> std::vector<std::reference_wrapper<variant const>> {
    auto output = std::vector<std::reference_wrapper<variant const>>();
    std::shared_lock _(mutex);
    output.reserve(groups.size());
    for (auto& [_, which] : groups) {
      output.emplace_back(which);
    }
    return output;
  }

  [[nodiscard]] auto size() const -> std::size_t {
    std::shared_lock _(mutex);
    return groups.size();
  }

 private:
  template <typename T, typename... A>
  auto make_variant(S name, A&&... args) -> variant& {
    // Return the variant if found, or insert a new variant into gourps.
    {
      std::shared_lock _(mutex);
      if (auto iter = groups.find(name); iter != groups.end()) {
        return iter->second;
      }
    }
    {
      // Avoid invoking std::make_shared if name already existed. Here we use a
      // empty shared pointer as dummy.
      using type = group<S, T>;
      auto dummy = std::shared_ptr<type>();

      std::unique_lock _(mutex);
      auto [iter, inserted] = groups.try_emplace(name, dummy);
      if (inserted) {
        iter->second = std::make_shared<type>(name, std::forward<A>(args)...);
      }
      return iter->second;
    }
  }
};

}  // namespace dipu::metrics::detail
