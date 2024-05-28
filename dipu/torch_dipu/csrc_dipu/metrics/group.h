#pragma once

#include <atomic>
#include <initializer_list>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "label.h"
#include "value.h"

namespace dipu::metrics::detail {

template <typename T>
struct shared_value {
  T value;
  std::atomic_bool use{};  // switch to std::atomic_flag in C++20
  std::atomic_uint count{};

  explicit shared_value(T const& source) noexcept : value(source) {}
  auto incref() noexcept -> void { count.fetch_add(1); }
  auto decref() noexcept -> void { count.fetch_sub(1); }
  auto touch() noexcept -> void { use.store(true, std::memory_order_release); }
  auto unused() const noexcept -> bool {
    return not use.load(std::memory_order_acquire) and count.load() == 0;
  }
};

}  // namespace dipu::metrics::detail

namespace dipu::metrics {

template <typename T, typename C>
class labeled_value;

template <typename T, typename C>
class group : public std::enable_shared_from_this<group<T, C>> {
  std::shared_mutex mutable mutex;
  std::basic_string<C> entry_name;
  std::basic_string<C> entry_type;
  std::basic_string<C> entry_description;
  T default_value;
  std::unordered_map<labelset<C>, detail::shared_value<T>> entries;

 public:
  using value_type = typename decltype(entries)::value_type;

  template <typename... U>
  explicit group(std::basic_string<C> name,  //
                 std::basic_string<C> desc,  //
                 U&&... args) noexcept
      : entry_name(std::move(name)),                 //
        entry_type(detail::value_name<T, C>::name),  //
        entry_description(std::move(desc)),          //
        default_value(std::forward<U>(args)...) {}

  auto name() const noexcept -> std::basic_string<C> const& {
    return entry_name;
  }

  auto type() const noexcept -> std::basic_string<C> const& {
    return entry_type;
  }

  auto description() const noexcept -> std::basic_string<C> const& {
    return entry_description;
  }

  [[nodiscard]] auto find_or_make(labelset<C> labels) -> labeled_value<T, C> {
    {
      std::shared_lock _(mutex);
      if (auto iter = entries.find(labels); iter != entries.end()) {
        return labeled_value<T, C>(this->shared_from_this(), *iter);
      }
    }
    {
      std::unique_lock _(mutex);
      auto [iter, done] = entries.try_emplace(std::move(labels), default_value);
      return labeled_value<T, C>(this->shared_from_this(), *iter);
    }
  }

  template <typename F>
  auto each(F f) {
    auto found_unused = false;
    {
      std::shared_lock _(mutex);
      for (auto& [key, value] : entries) {
        if (value.unused()) {
          found_unused = true;
        } else {
          f(key, value.value);
        }
      }
    }
    if (found_unused) {
      std::unique_lock _(mutex);
      for (auto iter = entries.begin(); iter != entries.end();) {
        if (iter->second.unused()) {
          iter = entries.erase(iter);
        } else {
          ++iter;
        }
      }
    }
  }
};

template <typename T, typename C>
class labeled_value : public detail::value_operation<labeled_value<T, C>> {
  using owner_type = group<T, C>;
  using value_type = typename owner_type::value_type;

  std::shared_ptr<owner_type> owner{};
  value_type* pointer{};

 private:
  friend struct detail::value_access<labeled_value>;
  auto access() const noexcept -> T const& { return pointer->second.value; }
  auto access() noexcept -> T& {
    pointer->second.touch();
    return pointer->second.value;
  }

  friend class group<T, C>;  // owner_type
  explicit labeled_value(std::shared_ptr<owner_type> owner,
                         value_type& reference) noexcept
      : owner(std::move(owner)), pointer(&reference) {
    increase_reference_count();
  }

 public:
  auto operator=(labeled_value const& other) noexcept -> labeled_value& {
    if (this != &other) {
      decrease_reference_count();
      owner = other.owner;
      pointer = other.pointer;
      increase_reference_count();
    }
    return *this;
  }

  auto operator=(labeled_value&& other) noexcept -> labeled_value& {
    if (this != &other) {
      decrease_reference_count();
      owner = std::move(other.owner);
      pointer = other.pointer;
      other.unset_nullptr();
    }
    return *this;
  }

  labeled_value(labeled_value const& other) noexcept
      : owner(other.owner), pointer(other.pointer) {
    increase_reference_count();
  }

  labeled_value(labeled_value&& other) noexcept
      : owner(std::move(other.owner)), pointer(other.pointer) {
    other.unset_nullptr();
  }

  ~labeled_value() {
    decrease_reference_count();
    unset_nullptr();
  }

  auto name() const noexcept -> std::basic_string<C> const& {
    return owner->name();
  }

  auto type() const noexcept -> std::basic_string<C> const& {
    return owner->type();
  }

  auto labels() const noexcept -> std::vector<label<C>> const& {
    return pointer->first.labels();
  }

  auto description() const noexcept -> std::basic_string<C> const& {
    return owner->description();
  }

  [[nodiscard]] auto with(labelset<C> const& labels) const -> labeled_value {
    return owner->find_or_make(pointer->first + labels);
  }

  [[nodiscard]] auto with(std::initializer_list<label<C>> labels) const
      -> labeled_value {
    return owner->find_or_make(pointer->first + labels);
  }

  [[nodiscard]] auto without(std::initializer_list<char const*> names) const
      -> labeled_value {
    return owner->find_or_make(pointer->first - names);
  }

 private:
  auto unset_nullptr() noexcept {
    owner = nullptr;
    pointer = nullptr;
  }

  auto decrease_reference_count() noexcept {
    if (pointer) {
      pointer->second.decref();
    }
  }

  auto increase_reference_count() noexcept {
    if (pointer) {
      pointer->second.incref();
    }
  }
};

}  // namespace dipu::metrics
