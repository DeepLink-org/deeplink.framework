#include <dlfcn.h>      // For dlsym, dlopen, dlerror
#include <stdexcept>    // For std::runtime_error
#include <string>       // For std::string
#include <iostream>
#include <map>

inline void* getOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName) {
    void* funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        std::cerr << "Warning: [" << __FILE__ << ":" << __LINE__ << "] " << __FUNCTION__ 
                  << ": dlsym " << apiName << " from " << libName << " failed, error: " << dlerror() << std::endl;
    }
    return funcAddr;
}

inline void* getOpApiLibHandler(const char* libName) {
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
    std::cerr << "Warning: " << __FILE__ << ":" << __LINE__ << " " << __FUNCTION__ 
          << " dlopen " << libName << " failed, error:" << dlerror() << std::endl;
    }
    return handler;
}

inline void* getOpApiFuncAddr(const char* apiName) {
    constexpr const char kOpApiLibName[] = "libpccl.so";
    static void* opApiHandler = getOpApiLibHandler(kOpApiLibName);
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return getOpApiFuncAddrInLib(opApiHandler, kOpApiLibName, apiName);
}

#define EXPAND(x) x
#define GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, NAME, ...) NAME
#define TYPE_PARAM(...) EXPAND(GET_MACRO(__VA_ARGS__, TYPE_PARAM_9, TYPE_PARAM_8, TYPE_PARAM_7, TYPE_PARAM_6, TYPE_PARAM_5, TYPE_PARAM_4, TYPE_PARAM_3, TYPE_PARAM_2, TYPE_PARAM_1)(__VA_ARGS__))
#define FORMAT_TYPE_PARAM(T, ...) T __VA_ARGS__
#define TYPE_PARAM_1(TP1) FORMAT_TYPE_PARAM TP1
#define TYPE_PARAM_2(TP1, TP2) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_1(TP2)
#define TYPE_PARAM_3(TP1, TP2, TP3) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_2(TP2, TP3)
#define TYPE_PARAM_4(TP1, TP2, TP3, TP4) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_3(TP2, TP3, TP4)
#define TYPE_PARAM_5(TP1, TP2, TP3, TP4, TP5) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_4(TP2, TP3, TP4, TP5)
#define TYPE_PARAM_6(TP1, TP2, TP3, TP4, TP5, TP6) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_5(TP2, TP3, TP4, TP5, TP6)
#define TYPE_PARAM_7(TP1, TP2, TP3, TP4, TP5, TP6, TP7) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_6(TP2, TP3, TP4, TP5, TP6, TP7)
#define TYPE_PARAM_8(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_7(TP2, TP3, TP4, TP5, TP6, TP7, TP8)
#define TYPE_PARAM_9(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9) FORMAT_TYPE_PARAM TP1, TYPE_PARAM_8(TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9)
#define PARAM(...) EXPAND(GET_MACRO(__VA_ARGS__, PARAM_9, PARAM_8, PARAM_7, PARAM_6, PARAM_5, PARAM_4, PARAM_3, PARAM_2, PARAM_1, PARAM_0)(__VA_ARGS__))
#define FORMAT_PARAM(T, ...) __VA_ARGS__
#define PARAM_1(TP1) FORMAT_PARAM TP1
#define PARAM_2(TP1, TP2) FORMAT_PARAM TP1, PARAM_1(TP2)
#define PARAM_3(TP1, TP2, TP3) FORMAT_PARAM TP1, PARAM_2(TP2, TP3)
#define PARAM_4(TP1, TP2, TP3, TP4) FORMAT_PARAM TP1, PARAM_3(TP2, TP3, TP4)
#define PARAM_5(TP1, TP2, TP3, TP4, TP5) FORMAT_PARAM TP1, PARAM_4(TP2, TP3, TP4, TP5
#define PARAM_6(TP1, TP2, TP3, TP4, TP5, TP6) FORMAT_PARAM TP1, PARAM_5(TP2, TP3, TP4, TP5, TP6)
#define PARAM_7(TP1, TP2, TP3, TP4, TP5, TP6, TP7) FORMAT_PARAM TP1, PARAM_6(TP2, TP3, TP4, TP5, TP6, TP7)
#define PARAM_8(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8) FORMAT_PARAM TP1, PARAM_7(TP2, TP3, TP4, TP5, TP6, TP7, TP8)
#define PARAM_9(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9) FORMAT_PARAM TP1, PARAM_8(TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9)

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)
