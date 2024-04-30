// Copyright (c) 2024, DeepLink.
#include <algorithm>
#include <ios>
#include <iostream>
#include <regex>

#include <c10/util/Exception.h>

namespace dipu {
namespace op_regex_match {
std::vector<std::regex> loadMatcher(const char* env_name,
                                    const char* config_name);
bool isOpMatch(const char* opname,
               const std::vector<std::regex>& regexMatchers);
extern const std::vector<std::regex> kFallbackMatchers;
extern const std::vector<std::regex> kAutocompareMatchers;
}  // namespace op_regex_match
}  // namespace dipu
