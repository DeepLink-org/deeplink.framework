#include <c10/util/Exception.h>

#include <algorithm>
#include <ios>
#include <iostream>
#include <regex>

namespace dipu {
std::vector<std::regex> loadMatcher(const char* env_name,
                                    const char* config_name);
bool whetherOpMatch(const char* opname, std::vector<std::regex> regexMatchers);
bool whetherGlobalAutocompare();
bool whetherAutoCompare(const char* opname,
                        std::vector<std::regex> autocompareMatchers);
}  // namespace dipu

extern const char* fallback_env_name;
extern const char* fallback_config_name;
extern std::vector<std::regex> fallbackMatchers;

extern const char* specified_autocompare_env_name;
extern const char* specified_autocompare_config_name;
extern std::vector<std::regex> autocompareMatchers;
