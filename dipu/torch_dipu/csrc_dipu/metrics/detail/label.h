#pragma once

#include <algorithm>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace dipu::metrics::detail {

// Label is a pair of string consists of name and value.
template <typename S /* string type */>
using label = std::pair<S, S>;

// An ordered list of unique labels.
template <typename S /* string type */>
class labelset {
  std::vector<label<S>> list;

 public:
  // Create a empty labelset.
  labelset() noexcept = default;

  // Create a labelset from lable lists.
  // Note: empty and duplicated labels will be ignored silently.
  labelset(std::initializer_list<label<S>> labels) : list(labels) {
    if (not list.empty()) {
      auto iter = std::remove_if(list.begin(), list.end(), is_empty);
      list.erase(iter, list.end());
      std::stable_sort(list.begin(), list.end());
      deduplicate_but_leave_the_last();
    }
  }

  auto operator==(labelset const& other) const noexcept -> bool {
    return this->list == other.list;
  }

  // Remove label (a.k.a. name-value pair) whose name is in that list.
  template <typename T>
  auto operator-=(std::initializer_list<T> that) -> labelset& {
    if (std::is_sorted(that.begin(), that.end())) {
      auto found = [that](auto& it) -> bool {
        auto iter = std::lower_bound(that.begin(), that.end(), it.first);
        return iter != that.end() && *iter == it.first;
      };
      list.erase(std::remove_if(list.begin(), list.end(), found), list.end());

    } else {
      auto found = [that](auto& it) -> bool {
        return std::find(that.begin(), that.end(), it.first) != that.end();
      };
      list.erase(std::remove_if(list.begin(), list.end(), found), list.end());
    }

    return *this;
  }

  // Append a list of label into current labels.
  //
  // Note: empty labels will be ignored, and the new label value will overwrite
  // the inner one if label keys are equal.
  auto operator+=(labelset const& other) -> labelset& {
    list.reserve(list.size() + other.list.size());
    auto iter = list.insert(list.end(), other.list.begin(), other.list.end());
    std::inplace_merge(list.begin(), iter, list.end());
    return deduplicate_but_leave_the_last();
  }

  // Append a list of label into current labels.
  //
  // Note: empty labels will be ignored, and the new label value will overwrite
  // the inner one if label keys are equal.
  auto operator+=(std::initializer_list<label<S>> that) -> labelset& {
    // [0] Remove empty and sort "that"
    list.reserve(list.size() + that.size());
    auto iter = list.insert(list.end(), that.begin(), that.end());
    list.erase(std::remove_if(iter, list.end(), is_empty), list.end());
    std::stable_sort(iter, list.end());
    // [1] Merge that into current list
    std::inplace_merge(list.begin(), iter, list.end());
    // [2] Remove duplicated but keeping the last one.
    return deduplicate_but_leave_the_last();
  }

  // Merge current labels and other labels. Labels will be removed if their
  // names are eqaul and the last one will be preserved.
  //
  // This method is a replacement to operator-, as initializer_list cannot be
  // used in binary operator.
  auto operator()(std::initializer_list<label<S>> that) const -> labelset {
    return clone() += that;
  }

  // Copy self.
  [[nodiscard]] auto clone() const -> labelset { return labelset(*this); }

  // Return the underlying list of labels.
  [[nodiscard]] auto labels() const noexcept -> std::vector<label<S>> const& {
    return list;
  }

 private:
  auto deduplicate_but_leave_the_last() -> labelset& {
    // Remove duplicated and only keep the last one. This allows a later
    // label *overwrite* previous one.
    auto riter = std::unique(list.rbegin(), list.rend(), is_same);
    list.erase(list.begin(), riter.base());
    return *this;
  }

  auto static is_empty(label<S> const& l) noexcept -> bool {
    return l.first.empty();
  }

  auto static is_same(label<S> const& l, label<S> const& r) noexcept -> bool {
    return l.first == r.first;
  }
};

// Custom deduction guide for constructor.
labelset(std::initializer_list<label<char const*>>)
    -> labelset<std::basic_string<char>>;

}  // namespace dipu::metrics::detail

template <typename S>
struct std::hash<dipu::metrics::detail::labelset<S>> {
  auto static combine(std::size_t seed, std::size_t value) -> std::size_t {
    // https://stackoverflow.com/questions/2590677
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    return seed ^ (value + 0x9e3779b9 + (seed << 6U) + (seed >> 2U));
  }

  std::size_t operator()(dipu::metrics::detail::labelset<S> const& l) const {
    auto hash = std::hash<S>{};
    auto& vec = l.labels();
    auto seed = vec.size();
    for (auto& [key, value] : vec) {
      seed = combine(seed, hash(key));
      seed = combine(seed, hash(value));
    }
    return seed;
  }
};
