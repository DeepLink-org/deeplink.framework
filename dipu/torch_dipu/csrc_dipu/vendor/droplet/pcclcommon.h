#include <dlfcn.h>  // For dlsym, dlopen, dlerror
#include <iostream>
#include <map>
#include <stdexcept>  // For std::runtime_error
#include <string>     // For std::string

inline void* getCommPcclFuncAddrInLib(void* handler, const char* libName,
                                      const char* apiName) {
  void* funcAddr = dlsym(handler, apiName);
  if (funcAddr == nullptr) {
    std::cerr << "Warning: [" << __FILE__ << ":" << __LINE__ << "] "
              << __FUNCTION__ << ": dlsym " << apiName << " from " << libName
              << " failed, error: " << dlerror() << std::endl;
  }
  return funcAddr;
}

inline void* getCommPcclLibHandler(const char* libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  if (handler == nullptr) {
    std::cerr << "Warning: " << __FILE__ << ":" << __LINE__ << " "
              << __FUNCTION__ << " dlopen " << libName
              << " failed, error:" << dlerror() << std::endl;
  }
  return handler;
}

inline void* getCommPcclFuncAddr(const char* apiName) {
  constexpr const char pcclLibName[] = "libpccl.so";
  constexpr const char pcclLibDependName[] = "libtangrt_shared.so";
  static void* pcclHandler = getCommPcclLibHandler(pcclLibName);
  if (pcclHandler == nullptr) {
    std::cerr << "Fallback " << apiName << " will be called" << std::endl;
    return nullptr;
  }
  return getCommPcclFuncAddrInLib(pcclHandler, pcclLibName, apiName);
}

#define EXPAND(x) x
#define DIPU_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, _9, NAME, ...) NAME
#define DIPU_TYPE_PARAM(...)                                                \
  EXPAND(DIPU_GET_MACRO(                                                    \
      __VA_ARGS__, DIPU_TYPE_PARAM_9, DIPU_TYPE_PARAM_8, DIPU_TYPE_PARAM_7, \
      DIPU_TYPE_PARAM_6, DIPU_TYPE_PARAM_5, DIPU_TYPE_PARAM_4,              \
      DIPU_TYPE_PARAM_3, DIPU_TYPE_PARAM_2, DIPU_TYPE_PARAM_1)(__VA_ARGS__))
#define DIPU_FORMAT_TYPE_PARAM(T, ...) T __VA_ARGS__
#define DIPU_TYPE_PARAM_1(TP1) DIPU_FORMAT_TYPE_PARAM TP1
#define DIPU_TYPE_PARAM_2(TP1, TP2) \
  DIPU_FORMAT_TYPE_PARAM TP1, DIPU_TYPE_PARAM_1(TP2)
#define DIPU_TYPE_PARAM_3(TP1, TP2, TP3) \
  DIPU_FORMAT_TYPE_PARAM TP1, DIPU_TYPE_PARAM_2(TP2, TP3)
#define DIPU_TYPE_PARAM_4(TP1, TP2, TP3, TP4) \
  DIPU_FORMAT_TYPE_PARAM TP1, DIPU_TYPE_PARAM_3(TP2, TP3, TP4)
#define DIPU_TYPE_PARAM_5(TP1, TP2, TP3, TP4, TP5) \
  DIPU_FORMAT_TYPE_PARAM TP1, DIPU_TYPE_PARAM_4(TP2, TP3, TP4, TP5)
#define DIPU_TYPE_PARAM_6(TP1, TP2, TP3, TP4, TP5, TP6) \
  DIPU_FORMAT_TYPE_PARAM TP1, DIPU_TYPE_PARAM_5(TP2, TP3, TP4, TP5, TP6)
#define DIPU_TYPE_PARAM_7(TP1, TP2, TP3, TP4, TP5, TP6, TP7) \
  DIPU_FORMAT_TYPE_PARAM TP1, DIPU_TYPE_PARAM_6(TP2, TP3, TP4, TP5, TP6, TP7)
#define DIPU_TYPE_PARAM_8(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8) \
  DIPU_FORMAT_TYPE_PARAM TP1,                                     \
      DIPU_TYPE_PARAM_7(TP2, TP3, TP4, TP5, TP6, TP7, TP8)
#define DIPU_TYPE_PARAM_9(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9) \
  DIPU_FORMAT_TYPE_PARAM TP1,                                          \
      DIPU_TYPE_PARAM_8(TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9)
#define DIPU_PARAM(...)                                                        \
  EXPAND(DIPU_GET_MACRO(__VA_ARGS__, DIPU_PARAM_9, DIPU_PARAM_8, DIPU_PARAM_7, \
                        DIPU_PARAM_6, DIPU_PARAM_5, DIPU_PARAM_4,              \
                        DIPU_PARAM_3, DIPU_PARAM_2,                            \
                        DIPU_PARAM_1)(__VA_ARGS__))
#define DIPU_FORMAT_PARAM(T, ...) __VA_ARGS__
#define DIPU_PARAM_1(TP1) DIPU_FORMAT_PARAM TP1
#define DIPU_PARAM_2(TP1, TP2) DIPU_FORMAT_PARAM TP1, DIPU_PARAM_1(TP2)
#define DIPU_PARAM_3(TP1, TP2, TP3) \
  DIPU_FORMAT_PARAM TP1, DIPU_PARAM_2(TP2, TP3)
#define DIPU_PARAM_4(TP1, TP2, TP3, TP4) \
  DIPU_FORMAT_PARAM TP1, DIPU_PARAM_3(TP2, TP3, TP4)
#define DIPU_PARAM_5(TP1, TP2, TP3, TP4, TP5) \
  DIPU_FORMAT_PARAM TP1, DIPU_PARAM_4(TP2, TP3, TP4, TP5)
#define DIPU_PARAM_6(TP1, TP2, TP3, TP4, TP5, TP6) \
  DIPU_FORMAT_PARAM TP1, DIPU_PARAM_5(TP2, TP3, TP4, TP5, TP6)
#define DIPU_PARAM_7(TP1, TP2, TP3, TP4, TP5, TP6, TP7) \
  DIPU_FORMAT_PARAM TP1, DIPU_PARAM_6(TP2, TP3, TP4, TP5, TP6, TP7)
#define DIPU_PARAM_8(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8) \
  DIPU_FORMAT_PARAM TP1, DIPU_PARAM_7(TP2, TP3, TP4, TP5, TP6, TP7, TP8)
#define DIPU_PARAM_9(TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9) \
  DIPU_FORMAT_PARAM TP1, DIPU_PARAM_8(TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9)

#define DIPU_CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) DIPU_CONCAT_IMPL(x, y)
