#ifndef DICP_ASCEND_GRAPH_UTILS_H
#define DICP_ASCEND_GRAPH_UTILS_H
#include <cctype>
#include <fstream>
#include <functional>
#include <half.hpp>
#include <iostream>
#include <json.hpp>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"
#include "all_ops.h"
#include "ascend_string.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "gnode.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#define FAILED -1
#define SUCCESS 0

using json = nlohmann::json;

#define DICP_CHECK_ABORT(condition, ...)                              \
    do {                                                              \
        if (!(condition)) {                                           \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
            std::abort();                                             \
        }                                                             \
    } while (0);

#define DICP_ASCEND_CHECK_NULLPTR_ABORT(ptr) DICP_CHECK_ABORT(ptr, "Variable is nullptr, please check.")

#define TRACK_GE(x)                                                         \
    do {                                                                    \
        static bool enable = std::getenv("DICP_NOT_TRACK_GE") == nullptr;   \
        if (enable) {                                                       \
            printf("[%s: %d]:%s\n", __FILE__, __LINE__, x);                 \
        }                                                                   \
    } while (0);

#define CALL_GE_FUNC(Expr)                                                                          \
    do {                                                                                            \
        auto ret = Expr;                                                                            \
        if (ret != SUCCESS) {                                                                       \
            TRACK_GE(#Expr);                                                                        \
            throw std::runtime_error("dicp call ge function failed.");                              \
        }                                                                                           \
    } while (0);

std::unordered_map<std::string, std::string> parse_json_to_map(const std::string& config_file) {
    std::ifstream f(config_file);
    json config_json = json::parse(f);
    std::unordered_map<std::string, std::string> conf;
    for (const auto& elem : config_json.items()) {
        if (elem.value().is_string()) {
            conf[elem.key()] = elem.value().get<std::string>();
        } else {
            throw std::runtime_error("in config file, json value is not string!");
        }
    }
    return conf;
}

#endif  // DICP_ASCEND_GRAPH_UTILS_H