#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <thread>
#include <type_traits>

namespace dipu::metrics::detail {

template <typename F, typename... T>
class defer final {
  F function;
  std::tuple<T...> arguments;
  int uncaughted;

 public:
  auto constexpr static noexcept_invokable =
      noexcept(std::apply(std::forward<F>(function), std::move(arguments)));

  // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
  explicit defer(F&& f, T&&... args) noexcept
      : function(std::forward<F>(f)),
        arguments(std::forward_as_tuple(std::forward<T>(args)...)),
        uncaughted(noexcept_invokable ? 0 : std::uncaught_exceptions()) {}

  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-destructor,performance-noexcept-destructor)
  ~defer() noexcept(noexcept_invokable) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if constexpr (noexcept_invokable) {
      std::apply(std::forward<F>(function), std::move(arguments));
    } else if (uncaughted == std::uncaught_exceptions()) {
      std::apply(std::forward<F>(function), std::move(arguments));
    } else {
      try {
        std::apply(std::forward<F>(function), std::move(arguments));
      } catch (std::exception& ex) {
        std::cerr << "exception during deferred call: " << ex.what() << "\n";
        throw ex;
      }
    }
  }

  defer(defer&&) = delete;
  defer(defer const&) = delete;
  defer& operator=(defer&&) = delete;
  defer& operator=(defer const&) = delete;
};

template <typename F, typename... T>
defer(F&&, T&&...) -> defer<F, T...>;

template <typename T, void (T::*release)(typename T::value_type&)>
class scoped final {
 public:
  using owner_type = T;
  using value_type = typename T::value_type;

 private:
  value_type* value{};
  owner_type* owner{};

  auto constexpr static inline is_release_noexcept =
      noexcept((owner->*release)(*value));

 public:
  explicit scoped(owner_type& owner, value_type& value) noexcept
      : value(&value), owner(&owner) {}

  scoped(scoped&& other) noexcept : value(other.value), owner(other.owner) {
    other.value = nullptr;
    other.owner = nullptr;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
  scoped& operator=(scoped&& other) noexcept(is_release_noexcept) {
    if (this == &other) {
      return *this;
    }
    if (value && owner) {
      (owner->*release)(*value);
    }
    value = other.value;
    owner = other.owner;
    other.value = nullptr;
    other.owner = nullptr;
    return *this;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-noexcept-destructor,performance-noexcept-destructor)
  ~scoped() noexcept(is_release_noexcept) {
    if (value && owner) {
      (owner->*release)(*value);
      value = nullptr;
      owner = nullptr;
    }
  }

  scoped(scoped const& other) = delete;
  scoped& operator=(scoped const& other) = delete;

  value_type& operator*() noexcept { return *value; }
  value_type* operator->() noexcept { return value; }
  value_type const& operator*() const noexcept { return *value; }
  value_type const* operator->() const noexcept { return value; }

  explicit operator value_type&() noexcept { return *value; }
  explicit operator value_type const&() const noexcept { return *value; }
};

template <typename T>
class double_buffered {
 public:
  using value_type = T;

 private:
  // Initial state:
  //
  // indexes[first]=0         indexes[second]=2
  // +-0---------+-1---------+-2---------+
  // | reference | reference | reference |
  // | value     | value     | value     |
  // +-----------+-----------+-----------+
  //
  // After flip:
  //
  // indexes[second]=0, indexes[first]=1
  // +-0---------+-1---------+-2---------+
  // | reference | reference | reference |
  // | value     | value     | value     |
  // +-----------+-----------+-----------+

  std::mutex flipping;
  std::array<std::atomic<uint32_t>, 2> indexes{0, 2};
  std::array<std::atomic<uint32_t>, 3> references{};
  std::array<value_type, 3> values{};

  enum which {
    first = 0,
    second = 1,
  };

  auto acquire_unsafe(which one) noexcept -> value_type& {
    auto index = indexes[one].fetch_add(4U) & 3U;
    references[index].fetch_add(1U);
    indexes[one].fetch_sub(4U);
    return values[index];
  }

  void release_unsafe(value_type& value) noexcept {
    auto index = &value - values.data();
    references[index].fetch_sub(1U);
  }

  auto move_next(which one) noexcept -> uint32_t {
    auto expected = indexes[one].load() & 3U;
    auto desired = (expected + 1U) % 3U;
    while (not indexes[one].compare_exchange_weak(expected, desired)) {
      expected &= 3U;
    }
    while (references[expected].load()) {  // busy wait
      std::this_thread::yield();
    }
    return expected;
  }

  auto static reset_default(value_type& value) -> void {
    static_assert(std::is_default_constructible_v<value_type>);
    if constexpr (std::is_trivially_default_constructible_v<value_type>) {
      std::memset(&value, 0, sizeof(value_type));

    } else {
      value.~value_type();
      new (&value) value_type{};
    }
  }

 public:
  // A RAII buffer guard.
  using borrowed = scoped<double_buffered, &double_buffered::release_unsafe>;

  // Borrow a buffer for producer.
  // Warning: `flip` will be blocked until borrowed got released.
  [[nodiscard]] auto producer() noexcept -> borrowed {
    return borrowed(*this, acquire_unsafe(first));
  }

  // Borrow a buffer for consumer.
  // Warning: `flip` will be blocked until borrowed got released.
  [[nodiscard]] auto consumer() noexcept -> borrowed {
    return borrowed(*this, acquire_unsafe(second));
  }

  // Wait until no borrowed existed and then swap buffers.
  template <auto f = reset_default>
  auto flip() -> void {
    std::scoped_lock _(flipping);
    move_next(first);
    f(values[move_next(second)]);
  }

  // Wait until no borrowed existed and then swap buffers.
  template <typename F>
  auto flip(F&& clear) -> void {
    std::scoped_lock _(flipping);
    move_next(first);
    std::forward<F>(clear)(values[move_next(second)]);
  }
};

}  // namespace dipu::metrics::detail
