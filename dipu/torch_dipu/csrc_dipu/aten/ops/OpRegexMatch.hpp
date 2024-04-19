// Copyright (c) 2024, DeepLink.
#include <algorithm>
#include <ios>
#include <iostream>
#include <regex>

#include <c10/util/Exception.h>

namespace dipu {
std::vector<std::regex> loadMatcher(const char* env_name,
                                    const char* config_name);
bool whetherOpMatch(const char* opname,
                    const std::vector<std::regex>& regexMatchers);
bool whetherGlobalAutocompare();
bool whetherAutoCompare(const char* opname,
                        const std::vector<std::regex>& autocompareMatchers);
}  // namespace dipu

extern const char* const fallback_env_name;
extern const char* const fallback_config_name;
extern const std::vector<std::regex> fallbackMatchers;

extern const char* const specified_autocompare_env_name;
extern const char* const specified_autocompare_config_name;
extern const std::vector<std::regex> autocompareMatchers;
