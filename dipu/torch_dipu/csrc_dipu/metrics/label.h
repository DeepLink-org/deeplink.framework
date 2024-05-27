#pragma once

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

namespace dipu::metrics {

template <typename C>
using label = std::pair<std::basic_string<C>, std::basic_string<C>>;

template <typename C>
class labelset {
 public:
  using label_t = label<C>;

 private:
  std::vector<label_t> list;

 public:
  labelset(std::initializer_list<label_t> labels) : list(labels) {
    if (not list.empty()) {
      auto iter = std::remove_if(list.begin(), list.end(), is_empty);
      list.erase(iter, list.end());
      std::stable_sort(list.begin(), list.end());
      auto riter = std::unique(list.rbegin(), list.rend(), is_same);
      list.erase(list.begin(), riter.base());
      list.shrink_to_fit();
    }
  }

  auto operator==(labelset const& other) const noexcept -> bool {
    return this->list == other.list;
  }

  explicit operator std::vector<label_t> const&() noexcept { return list; }

  explicit operator std::vector<label_t>&() noexcept { return list; }

  auto labels() const& -> std::vector<label_t> const& { return list; }

  auto operator-=(std::initializer_list<C const*> that) -> labelset& {
    auto exists = [that](auto& it) {
      return std::find(that.begin(), that.end(), it.first) != that.end();
    };
    list.erase(std::remove_if(list.begin(), list.end(), exists), list.end());
    return *this;
  }

  auto operator-=(std::initializer_list<label_t> that) -> labelset& {
    auto exists = [that](auto& it) {
      return std::find(that.begin(), that.end(), it) != that.end();
    };
    list.erase(std::remove_if(list.begin(), list.end(), exists), list.end());
    return *this;
  }

  auto operator-=(labelset const& other) -> labelset& {
    auto exists = [&list = other.list](auto& it) {
      auto iter = std::lower_bound(list.begin(), list.end(), it);
      return iter != list.end() && *iter == it;
    };
    list.erase(std::remove_if(list.begin(), list.end(), exists), list.end());
    return *this;
  }

  auto operator+=(std::initializer_list<label_t> that) -> labelset& {
    list.reserve(list.size() + that.size());
    auto first = list.insert(list.end(), that.begin(), that.end());
    auto last = std::remove_if(first, list.end(), is_empty);
    std::stable_sort(first, last);
    std::inplace_merge(list.begin(), first, last);
    list.erase(last, list.end());
    auto riter = std::unique(list.rbegin(), list.rend(), is_same);
    list.erase(list.begin(), riter.base());
    return *this;
  }

  auto operator+=(labelset const& other) -> labelset& {
    list.reserve(list.size() + other.list.size());
    auto iter = list.insert(list.end(), other.list.begin(), other.list.end());
    std::inplace_merge(list.begin(), iter, list.end());
    auto riter = std::unique(list.rbegin(), list.rend(), is_same);
    list.erase(list.begin(), riter.base());
    return *this;
  }

  auto operator-(std::initializer_list<char const*> that) const -> labelset {
    return labelset(*this) -= that;
  }

  auto operator-(labelset const& other) const -> labelset {
    return labelset(*this) -= other;
  }

  auto operator+(labelset const& other) const -> labelset {
    return labelset(*this) += other;
  }

 private:
  auto static is_empty(label_t const& l) noexcept -> bool {
    return l.first.empty();
  }

  auto static is_same(label_t const& l, label_t const& r) noexcept -> bool {
    return l.first == r.first;
  }
};

}  // namespace dipu::metrics

template <typename C>
struct std::hash<dipu::metrics::labelset<C>> {
  auto static combaine(std::size_t seed, std::size_t value) -> std::size_t {
    // https://stackoverflow.com/questions/2590677
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    return seed ^ (value + 0x9e3779b9 + (seed << 6U) + (seed >> 2U));
  }

  std::size_t operator()(dipu::metrics::labelset<C> const& l) const {
    auto hash = std::hash<std::basic_string<C>>{};
    auto& vec = l.labels();
    auto seed = vec.size();
    for (auto& [key, value] : vec) {
      seed = combaine(seed, hash(key));
      seed = combaine(seed, hash(value));
    }
    return seed;
  }
};
