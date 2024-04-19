// Copyright (c) 2024, DeepLink.
#include "OpRegexMatch.hpp"

#include <c10/util/Exception.h>

#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>
#include <regex>

// loadMatcher is used to get regex matcher from env_name and config
// fallback_env_name = "DIPU_FORCE_FALLBACK_OPS_LIST"; fallback_config_name =
// ".dipu_force_fallback_op_list.config" specified_autocompare_env_name =
// "SPECIFIED_AUTOCOMPARE_OPS_LIST"; specified_autocompare_config_name =
// ".specified_autocompare_op_list.config"

namespace dipu {
std::vector<std::regex> loadMatcher(const char* env_name,
                                    const char* config_name) {
  auto append = [](std::istream& input, std::vector<std::regex>& output) {
    auto constexpr separator = ',';

    auto line = std::string();
    while (std::getline(input, line)) {
      auto buffer = std::istringstream(line);
      auto pattern = std::string();
      while (std::getline(buffer, pattern, separator)) {
        if (pattern.empty()) {
          continue;
        }
        try {
          output.emplace_back(pattern);
        } catch (const std::regex_error& e) {
          TORCH_CHECK(false, e.what());
        }
      }
    }
  };

  auto list = std::vector<std::regex>();
  if (auto env = std::getenv(env_name)) {
    auto iss = std::istringstream(env);
    append(iss, list);
  }
  if (auto file = std::ifstream(config_name, std::ios::binary)) {
    append(file, list);
  }
  return list;
}

bool whetherOpMatch(const char* opname,
                    const std::vector<std::regex>& regexMatchers) {
  if (regexMatchers.empty() || opname == nullptr) {
    return false;
  }

  return std::any_of(
      regexMatchers.begin(), regexMatchers.end(),
      [&opname](auto& matcher) { return std::regex_match(opname, matcher); });
}

bool whetherGlobalAutocompare() {
  const char* globalAutocompare = std::getenv("USE_GLOBAL_AUTOCOMPARE");
  if (globalAutocompare == nullptr) {
    return false;
  }

  std::string globalAutocompareStr(globalAutocompare);
  for (char& c : globalAutocompareStr) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }

  if (globalAutocompareStr == "on") {
    return true;
  }
  if (globalAutocompareStr == "off") {
    return false;
  }

  std::cerr
      << "Error: USE_GLOBAL_AUTOCOMPARE can only be set to 'ON' or 'OFF'.\n";
  return false;
}

// Whether to enable AutoCompare is based on USE_GLOBAL_AUTOCOMPARE and
// SPECIFIED_AUTOCOMPARE_OPS_LIST
bool whetherAutoCompare(const char* opname,
                        const std::vector<std::regex>& autocompareMatchers) {
  // if USE_GLOBAL_AUTOCOMPARE is true, global autocompare is enabled regardless
  // the value of SPECIFIED_AUTOCOMPARE_OPS_LIST
  if (whetherGlobalAutocompare()) {
    return true;
  }
  // else if opname in SPECIFIED_AUTOCOMPARE_OPS_LIST, the specified op will be
  // autocomapred
  return whetherOpMatch(opname, autocompareMatchers);
}
}  // end of namespace dipu

const char* fallback_env_name = "DIPU_FORCE_FALLBACK_OPS_LIST";
const char* fallback_config_name = ".dipu_force_fallback_op_list.config";
const std::vector<std::regex> fallbackMatchers =
    dipu::loadMatcher(fallback_env_name, fallback_config_name);

const char* specified_autocompare_env_name = "SPECIFIED_AUTOCOMPARE_OPS_LIST";
const char* specified_autocompare_config_name =
    ".specified_autocompare_op_list.config";
const std::vector<std::regex> autocompareMatchers = dipu::loadMatcher(
    specified_autocompare_env_name, specified_autocompare_config_name);
